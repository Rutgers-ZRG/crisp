"""HPC job submission backend for batch relaxation.

Submits, monitors, and collects batch MLIP/DFT relaxation jobs on a remote
Slurm cluster.  Two SSH modes are supported:

- **Agent mode**: wraps caller-provided execute/upload/download functions.
- **Standalone mode**: wraps ``subprocess.run(["ssh", ...])`` / ``scp`` for
  use in scripts or notebooks.
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

logger = logging.getLogger(__name__)


class RelaxBackend(Enum):
    """Relaxation backend type."""
    MLIP = "mlip"
    MACE_MP = "mace_mp"
    MACE_FT = "mace_ft"
    MATTERSIM = "mattersim"
    MATTERSIM_REFORM = "mattersim_reform"
    VASP = "vasp"
    DFT = "dft"


@dataclass
class VASPConfig:
    """VASP-specific settings for DFT relaxation backend.

    Parameters
    ----------
    vasp_cmd : str
        Command to run VASP (e.g., ``"srun vasp_std"``).
    potcar_dir : str
        Directory with element subdirectories containing POTCAR files.
    encut : int
        Plane-wave energy cutoff (eV).
    ediff : float
        SCF convergence criterion (eV).
    ediffg : float
        Ionic convergence: negative = force (eV/A), positive = energy (eV).
    nsw : int
        Maximum ionic steps.
    isif : int
        Stress/relaxation mode (3 = full cell+ions).
    ibrion : int
        Ionic relaxation algorithm (2 = CG).
    kspacing : float
        K-point spacing (A^-1). If >0, uses KSPACING in INCAR.
        If <=0, generates Gamma-centered MP grid (target ~0.3 A^-1).
    npar : int
        NPAR parallelization tag.
    ismear : int
        Smearing method (0 = Gaussian).
    sigma : float
        Smearing width (eV).
    extra_incar : str
        Extra INCAR lines, semicolon-separated (e.g.,
        ``"METAGGA=R2SCAN;LUSE_VDW=.TRUE.;BPARAM=11.95;LASPH=.TRUE."``).
    ntasks : int
        Number of MPI tasks for VASP.
    module_cmds : str
        Shell commands to load modules needed by VASP (e.g., Intel MKL).
        Inserted into the Slurm script before conda activation.
    """
    vasp_cmd: str = "srun vasp_std"
    potcar_dir: str = ""
    encut: int = 520
    ediff: float = 1e-5
    ediffg: float = -0.03
    nsw: int = 200
    isif: int = 3
    ibrion: int = 2
    kspacing: float = 0.3
    npar: int = 4
    ismear: int = 0
    sigma: float = 0.1
    extra_incar: str = ""
    ntasks: int = 32
    module_cmds: str = "module load intel/17.0.4"


@dataclass
class RelaxJob:
    """Tracks state of a single HPC relaxation job."""
    job_id: Optional[str] = None       # Slurm job ID
    struct_idx: int = 0                # Index within batch
    generation: int = 0
    batch_label: str = ""
    remote_dir: str = ""               # Remote working directory
    status: str = "pending"            # pending / submitted / running / completed / failed
    atoms_in: Optional[Atoms] = None   # Input structure
    atoms_out: Optional[Atoms] = None  # Relaxed structure (after collection)
    energy: Optional[float] = None     # eV/atom
    enthalpy: Optional[float] = None   # eV/atom
    volume: Optional[float] = None     # A^3/atom
    retries: int = 0
    error: str = ""


# -- SSH adapter functions --------------------------------------------------

def make_subprocess_adapter(host: str, user: str = ""):
    """Create SSH/SCP callables using subprocess for standalone mode.

    Parameters
    ----------
    host : str
        SSH host (e.g., ``"cluster"`` or ``"user@cluster"``).
    user : str
        SSH user (prepended as ``user@host`` if provided).

    Returns
    -------
    ssh_exec, upload, download : callables
    """
    target = f"{user}@{host}" if user else host

    def ssh_exec(cmd: str, timeout: int = 120) -> str:
        result = subprocess.run(
            ["ssh", target, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SSH command failed (rc={result.returncode}): {result.stderr}"
            )
        return result.stdout.strip()

    def upload(local_path: str, remote_path: str) -> None:
        subprocess.run(
            ["scp", local_path, f"{target}:{remote_path}"],
            check=True, capture_output=True, timeout=120,
        )

    def download(remote_path: str, local_path: str) -> None:
        subprocess.run(
            ["scp", f"{target}:{remote_path}", local_path],
            check=True, capture_output=True, timeout=120,
        )

    return ssh_exec, upload, download


# -- Slurm script template --------------------------------------------------

_SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{gres_line}#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --ntasks={ntasks}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

{module_cmds}

cd $SLURM_SUBMIT_DIR
{run_cmd}
"""

# -- MLIP relax script template ---------------------------------------------

_MLIP_RELAX_SCRIPT = """\
#!/usr/bin/env python3
\"\"\"Self-contained MLIP relaxation script for HPC.\"\"\"
import sys, os
import numpy as np
from ase.io import read, write
from ase.optimize import LBFGS

try:
    from ase.filters import FrechetCellFilter as CellFilter
except ImportError:
    try:
        from ase.filters import ExpCellFilter as CellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter as CellFilter

from nequip.ase import NequIPCalculator

MODEL = "{model_path}"
DEVICE = "{device}"
PRESSURE_GPA = {pressure_GPa}
FMAX = {fmax}
STEPS = {max_steps}

_EV_A3_TO_GPA = 160.21766208

E_MIN = -20.0
E_MAX = 5.0

atoms = read("input.extxyz")
nat = len(atoms)
calc = NequIPCalculator.from_compiled_model(MODEL, device=DEVICE)
atoms.calc = calc

p_eV_A3 = PRESSURE_GPA / _EV_A3_TO_GPA
ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
opt = LBFGS(ecf, logfile="relax.log")

def check_energy():
    e = atoms.get_potential_energy() / nat
    if e < E_MIN or e > E_MAX:
        raise RuntimeError(f"Energy diverged: {{e:.4f}} eV/at")
opt.attach(check_energy, interval=10)

try:
    converged = opt.run(fmax=FMAX, steps=STEPS)
except RuntimeError as exc:
    print(f"ERROR: {{exc}}", file=sys.stderr)
    os._exit(1)

e_per_atom = atoms.get_potential_energy() / nat
vol = atoms.get_volume()
h_per_atom = (atoms.get_potential_energy() + p_eV_A3 * vol) / nat

if e_per_atom > E_MAX or e_per_atom < E_MIN:
    print(f"ERROR: unphysical energy {{e_per_atom:.4f}} eV/at", file=sys.stderr)
    os._exit(1)

atoms.info["energy_per_atom"] = e_per_atom
atoms.info["enthalpy_per_atom"] = h_per_atom
atoms.info["volume_per_atom"] = vol / nat
atoms.info["pressure_GPa"] = PRESSURE_GPA
atoms.info["converged"] = bool(converged)

write("relaxed.extxyz", atoms)
print(f"OK: E={{e_per_atom:.6f}} H={{h_per_atom:.6f}} V={{vol/nat:.3f}}")
"""

# -- MACE-MP relax script template --------------------------------------------

_MACE_MP_RELAX_SCRIPT = """\
#!/usr/bin/env python3
\"\"\"Self-contained MACE-MP relaxation script for HPC.\"\"\"
import sys, os
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE

try:
    from ase.filters import FrechetCellFilter as CellFilter
except ImportError:
    try:
        from ase.filters import ExpCellFilter as CellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter as CellFilter

from mace.calculators import mace_mp

MODEL_SIZE = "{model_size}"
DEVICE = "{device}"
PRESSURE_GPA = {pressure_GPa}
FMAX = {fmax}
STEPS = {max_steps}

_EV_A3_TO_GPA = 160.21766208

atoms = read("input.extxyz")
calc = mace_mp(model=MODEL_SIZE, device=DEVICE, default_dtype="float64")
atoms.calc = calc

p_eV_A3 = PRESSURE_GPA / _EV_A3_TO_GPA
ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
opt = FIRE(ecf, logfile="relax.log")
converged = opt.run(fmax=FMAX, steps=STEPS)

nat = len(atoms)
e_per_atom = atoms.get_potential_energy() / nat
vol = atoms.get_volume()
h_per_atom = (atoms.get_potential_energy() + p_eV_A3 * vol) / nat

if e_per_atom > 5.0 or e_per_atom < -20.0:
    print(f"ERROR: unphysical energy {{e_per_atom:.4f}} eV/at", file=sys.stderr)
    os._exit(1)

atoms.info["energy_per_atom"] = e_per_atom
atoms.info["enthalpy_per_atom"] = h_per_atom
atoms.info["volume_per_atom"] = vol / nat
atoms.info["pressure_GPa"] = PRESSURE_GPA
atoms.info["converged"] = bool(converged)

write("relaxed.extxyz", atoms)
print(f"OK: E={{e_per_atom:.6f}} H={{h_per_atom:.6f}} V={{vol/nat:.3f}}")
"""

# -- MACE fine-tuned relax script template ------------------------------------

_MACE_FT_RELAX_SCRIPT = """\
#!/usr/bin/env python3
\"\"\"Self-contained MACE fine-tuned model relaxation script for HPC.\"\"\"
import sys, os
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE

try:
    from ase.filters import FrechetCellFilter as CellFilter
except ImportError:
    try:
        from ase.filters import ExpCellFilter as CellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter as CellFilter

from mace.calculators import MACECalculator

MODEL_PATH = "{model_path}"
DEVICE = "{device}"
PRESSURE_GPA = {pressure_GPa}
FMAX = {fmax}
STEPS = {max_steps}

_EV_A3_TO_GPA = 160.21766208

# Energy bounds for early abort (eV/atom) — prevents MLIP divergence
E_MIN = -20.0
E_MAX = 5.0

atoms = read("input.extxyz")
nat = len(atoms)
calc = MACECalculator(model_paths=MODEL_PATH, device=DEVICE, default_dtype="float64")
atoms.calc = calc

p_eV_A3 = PRESSURE_GPA / _EV_A3_TO_GPA
ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
opt = FIRE(ecf, logfile="relax.log")

# Attach energy watchdog to abort on divergence
def check_energy():
    e = atoms.get_potential_energy() / nat
    if e < E_MIN or e > E_MAX:
        raise RuntimeError(f"Energy diverged: {{e:.4f}} eV/at")
opt.attach(check_energy, interval=10)

try:
    converged = opt.run(fmax=FMAX, steps=STEPS)
except RuntimeError as exc:
    print(f"ERROR: {{exc}}", file=sys.stderr)
    os._exit(1)

e_per_atom = atoms.get_potential_energy() / nat
vol = atoms.get_volume()
h_per_atom = (atoms.get_potential_energy() + p_eV_A3 * vol) / nat

if e_per_atom > E_MAX or e_per_atom < E_MIN:
    print(f"ERROR: unphysical energy {{e_per_atom:.4f}} eV/at", file=sys.stderr)
    os._exit(1)

atoms.info["energy_per_atom"] = e_per_atom
atoms.info["enthalpy_per_atom"] = h_per_atom
atoms.info["volume_per_atom"] = vol / nat
atoms.info["pressure_GPa"] = PRESSURE_GPA
atoms.info["converged"] = bool(converged)

write("relaxed.extxyz", atoms)
print(f"OK: E={{e_per_atom:.6f}} H={{h_per_atom:.6f}} V={{vol/nat:.3f}}")
"""

# -- MatterSim relax script template ------------------------------------------

_MATTERSIM_RELAX_SCRIPT = """\
#!/usr/bin/env python3
\"\"\"Self-contained MatterSim relaxation script for HPC.\"\"\"
import sys, os
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE

try:
    from ase.filters import FrechetCellFilter as CellFilter
except ImportError:
    try:
        from ase.filters import ExpCellFilter as CellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter as CellFilter

from mattersim.forcefield import MatterSimCalculator

DEVICE = "{device}"
PRESSURE_GPA = {pressure_GPa}
FMAX = {fmax}
STEPS = {max_steps}
MODEL_PATH = "{model_path}"

_EV_A3_TO_GPA = 160.21766208

E_MIN = -15.0
E_MAX = 5.0

atoms = read("input.extxyz")
nat = len(atoms)
if MODEL_PATH:
    calc = MatterSimCalculator(load_path=MODEL_PATH, device=DEVICE)
else:
    calc = MatterSimCalculator(device=DEVICE)
atoms.calc = calc

p_eV_A3 = PRESSURE_GPA / _EV_A3_TO_GPA
ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
opt = FIRE(ecf, logfile="relax.log")

def check_energy():
    e = atoms.get_potential_energy() / nat
    if e < E_MIN or e > E_MAX:
        raise RuntimeError(f"Energy diverged: {{e:.4f}} eV/at")
opt.attach(check_energy, interval=10)

try:
    converged = opt.run(fmax=FMAX, steps=STEPS)
except RuntimeError as exc:
    print(f"ERROR: {{exc}}", file=sys.stderr)
    os._exit(1)

e_per_atom = atoms.get_potential_energy() / nat
vol = atoms.get_volume()
h_per_atom = (atoms.get_potential_energy() + p_eV_A3 * vol) / nat

if e_per_atom > E_MAX or e_per_atom < E_MIN:
    print(f"ERROR: unphysical energy {{e_per_atom:.4f}} eV/at", file=sys.stderr)
    os._exit(1)

atoms.info["energy_per_atom"] = e_per_atom
atoms.info["enthalpy_per_atom"] = h_per_atom
atoms.info["volume_per_atom"] = vol / nat
atoms.info["pressure_GPa"] = PRESSURE_GPA
atoms.info["converged"] = bool(converged)

write("relaxed.extxyz", atoms)
print(f"OK: E={{e_per_atom:.6f}} H={{h_per_atom:.6f}} V={{vol/nat:.3f}}")
"""


# -- MatterSim + Reform relax script template ---------------------------------

_MATTERSIM_REFORM_RELAX_SCRIPT = """\
#!/usr/bin/env python3
\"\"\"MatterSim + Reform (MixedCalculator) relaxation script for HPC.

Uses MixedCalculator(MatterSim, Reform_Calculator) which starts with
pure Reform (symmetry-favoring) and transitions to pure MatterSim
(physical) over iter_max optimizer steps.

Reform energy = sum of pairwise FP distances within same atom type.
Minimized when same-type atoms have identical local environments,
i.e. high-symmetry structures.
\"\"\"
import sys, os
import numpy as np
from ase.io import read, write
from ase.optimize import LBFGS

try:
    from ase.filters import FrechetCellFilter as CellFilter
except ImportError:
    try:
        from ase.filters import ExpCellFilter as CellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter as CellFilter

from mattersim.forcefield import MatterSimCalculator
from reformpy.calculator import Reform_Calculator
from reformpy.mixing import MixedCalculator

DEVICE = "{device}"
PRESSURE_GPA = {pressure_GPa}
FMAX = {fmax}
STEPS = {max_steps}

# Reform parameters
REFORM_CUTOFF = {reform_cutoff}
REFORM_NX = {reform_nx}
REFORM_ITER_MAX = {reform_iter_max}
REFORM_SCHEME = "{reform_scheme}"
REFORM_ZNUCL = {reform_znucl}
REFORM_NTYP = {reform_ntyp}
MODEL_PATH = "{model_path}"

_EV_A3_TO_GPA = 160.21766208

E_MIN = -15.0
E_MAX = 5.0

atoms = read("input.extxyz")
nat = len(atoms)

# Build MixedCalculator: starts pure Reform, transitions to pure MatterSim
if MODEL_PATH:
    ms_calc = MatterSimCalculator(load_path=MODEL_PATH, device=DEVICE)
else:
    ms_calc = MatterSimCalculator(device=DEVICE)

# Check if cell is large enough for Reform cutoff
cell_lengths = atoms.cell.lengths()
cell_ok = all(l >= 2 * REFORM_CUTOFF for l in cell_lengths)

if cell_ok:
    try:
        rf_calc = Reform_Calculator(
            ntyp=REFORM_NTYP, nx=REFORM_NX, cutoff=REFORM_CUTOFF,
            znucl=REFORM_ZNUCL, lmax=0, stress_mode="analytical",
        )
        calc = MixedCalculator(ms_calc, rf_calc,
                               iter_max=REFORM_ITER_MAX,
                               scheme=REFORM_SCHEME)
    except Exception as exc:
        print(f"WARNING: Reform init failed ({{exc}}), using pure MatterSim",
              file=sys.stderr)
        calc = ms_calc
else:
    print(f"WARNING: Cell too small for Reform (min={{min(cell_lengths):.2f}} < {{2*REFORM_CUTOFF:.1f}}), "
          f"using pure MatterSim", file=sys.stderr)
    calc = ms_calc

atoms.calc = calc

p_eV_A3 = PRESSURE_GPA / _EV_A3_TO_GPA
ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
opt = LBFGS(ecf, logfile="relax.log")

def check_energy():
    e = atoms.get_potential_energy() / nat
    if e < E_MIN or e > E_MAX:
        raise RuntimeError(f"Energy diverged: {{e:.4f}} eV/at")
opt.attach(check_energy, interval=10)

try:
    converged = opt.run(fmax=FMAX, steps=STEPS)
except RuntimeError as exc:
    print(f"ERROR: {{exc}}", file=sys.stderr)
    os._exit(1)

e_per_atom = atoms.get_potential_energy() / nat
vol = atoms.get_volume()
h_per_atom = (atoms.get_potential_energy() + p_eV_A3 * vol) / nat

if e_per_atom > E_MAX or e_per_atom < E_MIN:
    print(f"ERROR: unphysical energy {{e_per_atom:.4f}} eV/at", file=sys.stderr)
    os._exit(1)

atoms.info["energy_per_atom"] = e_per_atom
atoms.info["enthalpy_per_atom"] = h_per_atom
atoms.info["volume_per_atom"] = vol / nat
atoms.info["pressure_GPa"] = PRESSURE_GPA
atoms.info["converged"] = bool(converged)

write("relaxed.extxyz", atoms)
print(f"OK: E={{e_per_atom:.6f}} H={{h_per_atom:.6f}} V={{vol/nat:.3f}}")
"""


# -- VASP relax script template -----------------------------------------------

_VASP_PREP_SCRIPT = """\
#!/usr/bin/env python3
\"\"\"VASP input preparation for CRISP HPC backend.

Reads input.extxyz -> writes POSCAR, INCAR, KPOINTS, POTCAR.
\"\"\"
import os, sys
import numpy as np
from ase.io import read
from ase.io.vasp import write_vasp

PRESSURE_GPA = {pressure_GPa}
ENCUT = {encut}
EDIFF = {ediff}
EDIFFG = {ediffg}
NSW = {nsw}
ISIF = {isif}
IBRION = {ibrion}
KSPACING = {kspacing}
NPAR = {npar}
ISMEAR = {ismear}
SIGMA = {sigma}
POTCAR_DIR = "{potcar_dir}"
EXTRA_INCAR = "{extra_incar}"

# ---- 1. Read input ----
atoms = read("input.extxyz")
nat = len(atoms)

# ---- 2. Write POSCAR (sorted by atomic number) ----
write_vasp("POSCAR", atoms, sort=True, vasp5=True, direct=True)

# ---- 3. Determine element order from POSCAR ----
with open("POSCAR") as f:
    poscar_lines = f.readlines()
elements = poscar_lines[5].split()
counts = [int(x) for x in poscar_lines[6].split()]
print("Elements: " + " ".join(elements) + "  counts: " + " ".join(str(c) for c in counts))

# ---- 4. Build POTCAR ----
with open("POTCAR", "w") as fout:
    for el in elements:
        pp = os.path.join(POTCAR_DIR, el, "POTCAR")
        if not os.path.exists(pp):
            for suffix in ["_sv", "_pv", "_GW", ""]:
                alt = os.path.join(POTCAR_DIR, el + suffix, "POTCAR")
                if os.path.exists(alt):
                    pp = alt
                    break
        if not os.path.exists(pp):
            print("ERROR: POTCAR not found for " + el, file=sys.stderr)
            os._exit(1)
        with open(pp) as fin:
            fout.write(fin.read())
        print("POTCAR: " + el + " <- " + pp)

# ---- 5. Write INCAR ----
pstress_kbar = PRESSURE_GPA * 10.0
incar = []
incar.append("SYSTEM = CRISP")
incar.append("ISTART = 0")
incar.append("ICHARG = 2")
incar.append("ENCUT = " + str(ENCUT))
incar.append("EDIFF = " + str(EDIFF))
incar.append("EDIFFG = " + str(EDIFFG))
incar.append("NSW = " + str(NSW))
incar.append("ISIF = " + str(ISIF))
incar.append("IBRION = " + str(IBRION))
incar.append("POTIM = 0.5")
incar.append("PSTRESS = " + str(pstress_kbar))
incar.append("ISMEAR = " + str(ISMEAR))
incar.append("SIGMA = " + str(SIGMA))
incar.append("LREAL = .FALSE.")
incar.append("PREC = Accurate")
incar.append("NPAR = " + str(NPAR))
incar.append("LWAVE = .FALSE.")
incar.append("LCHARG = .FALSE.")
if KSPACING > 0:
    incar.append("KSPACING = " + str(KSPACING))
if EXTRA_INCAR:
    for item in EXTRA_INCAR.split(";"):
        item = item.strip()
        if item:
            incar.append(item)
with open("INCAR", "w") as f:
    for line in incar:
        f.write(line + "\\n")

# ---- 6. Write KPOINTS (if not using KSPACING) ----
if KSPACING <= 0:
    cell = np.array(atoms.cell)
    recip = 2.0 * np.pi * np.linalg.inv(cell).T
    b_lengths = np.linalg.norm(recip, axis=1)
    kpts = [max(1, int(round(bl / 0.3))) for bl in b_lengths]
    total = kpts[0] * kpts[1] * kpts[2]
    if total > 500:
        scale = (500.0 / total) ** (1.0 / 3.0)
        kpts = [max(1, int(round(k * scale))) for k in kpts]
    with open("KPOINTS", "w") as f:
        f.write("Automatic\\n")
        f.write("0\\n")
        f.write("Gamma\\n")
        f.write(str(kpts[0]) + " " + str(kpts[1]) + " " + str(kpts[2]) + "\\n")
        f.write("0 0 0\\n")
    print("KPOINTS: " + str(kpts[0]) + "x" + str(kpts[1]) + "x" + str(kpts[2]))

print("VASP inputs ready")
"""


_VASP_PARSE_SCRIPT = """\
#!/usr/bin/env python3
\"\"\"VASP output parser for CRISP HPC backend.

Parses CONTCAR + OUTCAR -> relaxed.extxyz.
\"\"\"
import os, sys
from ase.io import read, write

PRESSURE_GPA = {pressure_GPa}
_EV_A3_TO_GPA = 160.21766208

# ---- Read input to get nat ----
atoms_in = read("input.extxyz")
nat = len(atoms_in)

# ---- Parse output ----
if not os.path.exists("CONTCAR") or os.path.getsize("CONTCAR") < 10:
    print("ERROR: CONTCAR missing or empty", file=sys.stderr)
    os._exit(1)

atoms_out = read("CONTCAR", format="vasp")

energy = None
converged = False
if os.path.exists("OUTCAR"):
    with open("OUTCAR") as f:
        for line in f:
            if "free  energy   TOTEN" in line:
                try:
                    energy = float(line.split()[-2])
                except (ValueError, IndexError):
                    pass
            if "reached required accuracy" in line:
                converged = True

if energy is None:
    print("ERROR: could not parse energy from OUTCAR", file=sys.stderr)
    os._exit(1)

e_per_atom = energy / nat
p_eV_A3 = PRESSURE_GPA / _EV_A3_TO_GPA
vol = atoms_out.get_volume()
h_per_atom = (energy + p_eV_A3 * vol) / nat

atoms_out.info["energy_per_atom"] = e_per_atom
atoms_out.info["enthalpy_per_atom"] = h_per_atom
atoms_out.info["volume_per_atom"] = vol / nat
atoms_out.info["pressure_GPa"] = PRESSURE_GPA
atoms_out.info["converged"] = converged

write("relaxed.extxyz", atoms_out)
print("OK: E=" + str(round(e_per_atom, 6)) + " H=" + str(round(h_per_atom, 6))
      + " V=" + str(round(vol / nat, 3)) + " converged=" + str(converged))
"""


class HPCRelaxer:
    """Submit, monitor, and collect batch relaxation jobs on HPC.

    Parameters
    ----------
    ssh_exec : callable
        ``ssh_exec(cmd, timeout=120) -> stdout_str``
    upload : callable
        ``upload(local_path, remote_path) -> None``
    download : callable
        ``download(remote_path, local_path) -> None``
    remote_base : str
        Base directory on HPC for job files (e.g., ``/scratch/$USER/crisp_runs``).
    backend : RelaxBackend
        MLIP or DFT relaxation.
    model_path : str
        Path to MLIP model on HPC (for MLIP backend).
    conda_env : str
        Conda environment to activate on HPC.
    pressure_GPa : float
        External pressure for enthalpy.
    partition : str
        Slurm partition.
    gres : str
        Slurm GPU resource spec.
    time_limit : str
        Slurm time limit.
    mem : str
        Slurm memory limit.
    fmax : float
        Force convergence threshold.
    max_steps : int
        Maximum relaxation steps.
    poll_interval : float
        Seconds between squeue polls.
    max_retries : int
        Retry count for failed jobs.
    """

    def __init__(
        self,
        ssh_exec: Callable,
        upload: Callable,
        download: Callable,
        remote_base: str,
        backend: RelaxBackend = RelaxBackend.MLIP,
        model_path: str = "",
        conda_env: str = "nequip",
        conda_path: str = "conda",
        pressure_GPa: float = 0.0,
        partition: str = "gpu",
        gres: str = "gpu:1",
        time_limit: str = "00:30:00",
        mem: str = "16GB",
        fmax: float = 0.05,
        max_steps: int = 200,
        device: str = "cuda",
        model_size: str = "medium",
        poll_interval: float = 30.0,
        max_retries: int = 1,
        # Reform parameters (for MATTERSIM_REFORM backend)
        reform_cutoff: float = 3.5,
        reform_nx: int = 100,
        reform_iter_max: int = 30,
        reform_scheme: str = "cosine",
        reform_znucl: Optional[List[int]] = None,
        reform_ntyp: int = 1,
        # VASP parameters
        vasp_config: Optional['VASPConfig'] = None,
        ntasks: int = 1,
    ):
        self.ssh_exec = ssh_exec
        self.upload = upload
        self.download = download
        self.remote_base = remote_base.rstrip("/")
        self.backend = backend
        self.model_path = model_path
        self.conda_env = conda_env
        self.conda_path = conda_path
        self.pressure_GPa = pressure_GPa
        self.partition = partition
        self.gres = gres
        self.time_limit = time_limit
        self.mem = mem
        self.fmax = fmax
        self.max_steps = max_steps
        self.device = device
        self.model_size = model_size
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        # Reform
        self.reform_cutoff = reform_cutoff
        self.reform_nx = reform_nx
        self.reform_iter_max = reform_iter_max
        self.reform_scheme = reform_scheme
        self.reform_znucl = reform_znucl or [5]
        self.reform_ntyp = reform_ntyp
        # VASP
        self.vasp_config = vasp_config or VASPConfig()
        self.ntasks = ntasks

    def submit_batch(self, structures: List[Atoms], generation: int,
                     batch_label: str = "") -> List[RelaxJob]:
        """Submit a batch of structures for relaxation on HPC.

        Parameters
        ----------
        structures : list of Atoms
            Unrelaxed structures.
        generation : int
            Search generation number.
        batch_label : str
            Label for this batch (default: ``gen_NNNN``).

        Returns
        -------
        list of RelaxJob
            Jobs with ``status='submitted'`` and ``job_id`` set.
        """
        if not batch_label:
            batch_label = f"gen_{generation:04d}"

        batch_remote = f"{self.remote_base}/{batch_label}"
        self.ssh_exec(f"mkdir -p {batch_remote}")

        jobs = []
        for idx, atoms in enumerate(structures):
            job = RelaxJob(
                struct_idx=idx,
                generation=generation,
                batch_label=batch_label,
                remote_dir=f"{batch_remote}/struct_{idx:04d}",
                atoms_in=atoms.copy(),
            )
            try:
                self._submit_single(job)
            except Exception as exc:
                logger.warning("Failed to submit struct_%04d: %s", idx, exc)
                job.status = "failed"
                job.error = str(exc)
            jobs.append(job)

        n_submitted = sum(1 for j in jobs if j.status == "submitted")
        logger.info("Submitted %d / %d jobs for %s",
                     n_submitted, len(structures), batch_label)
        return jobs

    def _submit_single(self, job: RelaxJob) -> None:
        """Upload files and sbatch a single job."""
        self.ssh_exec(f"mkdir -p {job.remote_dir}")

        # Write input structure locally, upload
        with tempfile.NamedTemporaryFile(suffix=".extxyz", delete=False,
                                         mode="w") as f:
            tmpfile = f.name
        try:
            ase_write(tmpfile, job.atoms_in, format="extxyz")
            self.upload(tmpfile, f"{job.remote_dir}/input.extxyz")
        finally:
            os.unlink(tmpfile)

        # Generate and upload relax script
        if self.backend == RelaxBackend.MLIP:
            relax_py = _MLIP_RELAX_SCRIPT.format(
                model_path=self.model_path,
                device=self.device,
                pressure_GPa=self.pressure_GPa,
                fmax=self.fmax,
                max_steps=self.max_steps,
            )
            module_cmds = f"eval \"$({self.conda_path} shell.bash hook 2>/dev/null)\"\nconda activate {self.conda_env}"
            run_cmd = "python relax.py"
        elif self.backend == RelaxBackend.MACE_MP:
            relax_py = _MACE_MP_RELAX_SCRIPT.format(
                model_size=self.model_size,
                device=self.device,
                pressure_GPa=self.pressure_GPa,
                fmax=self.fmax,
                max_steps=self.max_steps,
            )
            module_cmds = f"eval \"$({self.conda_path} shell.bash hook 2>/dev/null)\"\nconda activate {self.conda_env}"
            run_cmd = "python relax.py"
        elif self.backend == RelaxBackend.MACE_FT:
            relax_py = _MACE_FT_RELAX_SCRIPT.format(
                model_path=self.model_path,
                device=self.device,
                pressure_GPa=self.pressure_GPa,
                fmax=self.fmax,
                max_steps=self.max_steps,
            )
            module_cmds = f"eval \"$({self.conda_path} shell.bash hook 2>/dev/null)\"\nconda activate {self.conda_env}"
            run_cmd = "python relax.py"
        elif self.backend == RelaxBackend.MATTERSIM:
            relax_py = _MATTERSIM_RELAX_SCRIPT.format(
                device=self.device,
                pressure_GPa=self.pressure_GPa,
                fmax=self.fmax,
                max_steps=self.max_steps,
                model_path=self.model_path,
            )
            module_cmds = f"eval \"$({self.conda_path} shell.bash hook 2>/dev/null)\"\nconda activate {self.conda_env}"
            run_cmd = "python relax.py"
        elif self.backend == RelaxBackend.MATTERSIM_REFORM:
            relax_py = _MATTERSIM_REFORM_RELAX_SCRIPT.format(
                device=self.device,
                pressure_GPa=self.pressure_GPa,
                fmax=self.fmax,
                max_steps=self.max_steps,
                reform_cutoff=self.reform_cutoff,
                reform_nx=self.reform_nx,
                reform_iter_max=self.reform_iter_max,
                reform_scheme=self.reform_scheme,
                reform_znucl=self.reform_znucl,
                reform_ntyp=self.reform_ntyp,
                model_path=self.model_path,
            )
            module_cmds = f"eval \"$({self.conda_path} shell.bash hook 2>/dev/null)\"\nconda activate {self.conda_env}"
            run_cmd = "python relax.py"
        elif self.backend == RelaxBackend.VASP:
            vc = self.vasp_config
            vasp_fmt_args = dict(
                pressure_GPa=self.pressure_GPa,
                encut=vc.encut,
                ediff=vc.ediff,
                ediffg=vc.ediffg,
                nsw=vc.nsw,
                isif=vc.isif,
                ibrion=vc.ibrion,
                kspacing=vc.kspacing,
                npar=vc.npar,
                ismear=vc.ismear,
                sigma=vc.sigma,
                potcar_dir=vc.potcar_dir,
                extra_incar=vc.extra_incar,
            )
            relax_py = _VASP_PREP_SCRIPT.format(**vasp_fmt_args)
            parse_py = _VASP_PARSE_SCRIPT.format(pressure_GPa=self.pressure_GPa)
            # VASP needs module loads (e.g., Intel MKL) before conda
            vasp_modules = vc.module_cmds + "\n" if vc.module_cmds else ""
            module_cmds = f"{vasp_modules}eval \"$({self.conda_path} shell.bash hook 2>/dev/null)\"\nconda activate {self.conda_env}"
            # Three-phase: prep inputs → run VASP via srun → parse outputs
            # srun must be called directly from the shell (not from Python)
            # to get proper PMI/MPI initialization within the Slurm allocation
            run_cmd = f"python vasp_prep.py && {vc.vasp_cmd} && python vasp_parse.py"
        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented")

        gres_line = f"#SBATCH --gres={self.gres}\n" if self.gres else ""
        # VASP overrides ntasks from its config
        ntasks = self.vasp_config.ntasks if self.backend == RelaxBackend.VASP else self.ntasks
        sbatch_sh = _SLURM_TEMPLATE.format(
            job_name=f"crisp_{job.batch_label}_{job.struct_idx:04d}",
            partition=self.partition,
            gres_line=gres_line,
            time_limit=self.time_limit,
            mem=self.mem,
            ntasks=ntasks,
            module_cmds=module_cmds,
            run_cmd=run_cmd,
        )

        # Upload scripts
        upload_files = [("sbp.sh", sbatch_sh)]
        if self.backend == RelaxBackend.VASP:
            upload_files.append(("vasp_prep.py", relax_py))
            upload_files.append(("vasp_parse.py", parse_py))
        else:
            upload_files.append(("relax.py", relax_py))
        for fname, content in upload_files:
            with tempfile.NamedTemporaryFile(suffix=f"_{fname}", delete=False,
                                             mode="w") as f:
                f.write(content)
                tmpfile = f.name
            try:
                self.upload(tmpfile, f"{job.remote_dir}/{fname}")
            finally:
                os.unlink(tmpfile)

        # sbatch
        out = self.ssh_exec(f"cd {job.remote_dir} && sbatch sbp.sh")
        # Parse "Submitted batch job 12345678"
        parts = out.strip().split()
        job.job_id = parts[-1]
        job.status = "submitted"
        logger.debug("Submitted job %s for struct_%04d", job.job_id, job.struct_idx)

    def poll_and_collect(self, jobs: List[RelaxJob]) -> List[RelaxJob]:
        """Poll HPC until all jobs complete, then collect results.

        Modifies jobs in-place: sets ``status``, ``atoms_out``, ``energy``,
        ``enthalpy``, ``volume`` for completed jobs.

        Returns the same list for chaining.
        """
        pending = [j for j in jobs if j.status == "submitted"]
        if not pending:
            return jobs

        job_ids = {j.job_id: j for j in pending if j.job_id}
        logger.info("Polling %d jobs (poll_interval=%.0fs)...",
                     len(pending), self.poll_interval)

        while job_ids:
            time.sleep(self.poll_interval)
            # Check which jobs are still in the queue
            ids_str = ",".join(job_ids.keys())
            try:
                out = self.ssh_exec(f"squeue -j {ids_str} -h -o '%i %T' 2>/dev/null || true")
            except Exception:
                out = ""

            still_running = {}
            for line in out.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    jid = parts[0]
                    state = parts[1]
                    if jid in job_ids:
                        still_running[jid] = state

            # Jobs that disappeared from squeue are done (or failed)
            finished_ids = set(job_ids.keys()) - set(still_running.keys())
            for jid in finished_ids:
                job = job_ids.pop(jid)
                self._collect_single(job)

            n_remain = len(job_ids)
            if n_remain > 0:
                states = list(still_running.values())
                logger.info("  %d jobs remaining (%s)", n_remain,
                            ", ".join(f"{s}:{states.count(s)}" for s in set(states)))

        # Retry failed jobs
        failed = [j for j in jobs if j.status == "failed" and j.retries < self.max_retries]
        if failed:
            logger.info("Retrying %d failed jobs...", len(failed))
            for job in failed:
                job.retries += 1
                job.status = "pending"
                job.error = ""
                try:
                    self._submit_single(job)
                except Exception as exc:
                    job.status = "failed"
                    job.error = str(exc)

            retry_submitted = [j for j in failed if j.status == "submitted"]
            if retry_submitted:
                self.poll_and_collect(jobs)

        n_ok = sum(1 for j in jobs if j.status == "completed")
        n_fail = sum(1 for j in jobs if j.status == "failed")
        logger.info("Batch done: %d completed, %d failed", n_ok, n_fail)
        return jobs

    def _collect_single(self, job: RelaxJob) -> None:
        """Download and parse results for a single completed job."""
        # Check if relaxed.extxyz exists
        try:
            check = self.ssh_exec(
                f"test -f {job.remote_dir}/relaxed.extxyz && echo OK || echo MISSING"
            )
        except Exception:
            check = "MISSING"

        if "OK" not in check:
            # Check sacct for failure reason
            try:
                sacct = self.ssh_exec(
                    f"sacct -j {job.job_id} --format=State --noheader -P 2>/dev/null || true"
                )
                job.error = f"No output; sacct: {sacct.strip()}"
            except Exception:
                job.error = "No relaxed.extxyz and sacct failed"
            job.status = "failed"
            logger.warning("Job %s failed: %s", job.job_id, job.error)
            return

        # Download relaxed structure
        with tempfile.NamedTemporaryFile(suffix=".extxyz", delete=False) as f:
            tmpfile = f.name
        try:
            self.download(f"{job.remote_dir}/relaxed.extxyz", tmpfile)
            atoms = ase_read(tmpfile, format="extxyz")
        except Exception as exc:
            job.status = "failed"
            job.error = f"Download/parse failed: {exc}"
            logger.warning("Job %s collection failed: %s", job.job_id, exc)
            return
        finally:
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)

        job.atoms_out = atoms
        job.energy = atoms.info.get("energy_per_atom")
        job.enthalpy = atoms.info.get("enthalpy_per_atom")
        job.volume = atoms.info.get("volume_per_atom")
        job.status = "completed"
        logger.debug("Collected job %s: E=%.4f H=%.4f",
                      job.job_id, job.energy or 0, job.enthalpy or 0)
