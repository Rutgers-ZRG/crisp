"""Benchmark system registry with literature reference structures.

Each :class:`SystemSpec` defines a fixed-composition CSP problem with known
ground truth. Reference builders produce literature structures; they need to
be *basin-correct* (they are re-relaxed on each potential during
calibration), not crystallographically perfect.

Ground truth is per-potential: calibration relaxes every listed polymorph on
the potential and the lowest-enthalpy one becomes the target. If that is not
the literature ``expected_gs``, the combo is flagged ``gt_mismatch`` (still
usable — the search must find the potential's own ground state).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from ase import Atoms
from ase.spacegroup import crystal


@dataclass
class SystemSpec:
    name: str
    composition: Dict[str, int]
    pressure_GPa: float
    difficulty: str                       # 'easy' | 'medium' | 'hard'
    refs: Dict[str, Callable[[], Atoms]]  # polymorph name -> builder
    expected_gs: str   # expected GS among refs (must tile composition)
    vol_per_atom_range: tuple
    min_dist_ang: float = 1.5
    dup_threshold: float = 0.03
    success_dfp: float = 0.05
    success_dH_meV: float = 5.0
    max_generations: int = 25
    n_random: int = 15
    n_mutants: int = 5
    fp_cutoff: float = 5.0
    fp_natx: int = 150

    @property
    def n_atoms(self) -> int:
        return sum(self.composition.values())


# ----------------------------------------------------------------------
# Reference builders (literature structures)
# ----------------------------------------------------------------------

def si_diamond() -> Atoms:
    """Si diamond, Fd-3m (227), a=5.431 (conv. 8 atoms)."""
    return crystal('Si', [(0.0, 0.0, 0.0)], spacegroup=227,
                   cellpar=[5.431, 5.431, 5.431, 90, 90, 90])


def si_beta_tin() -> Atoms:
    """Si beta-tin, I4_1/amd (141), high-pressure phase (hull context)."""
    return crystal('Si', [(0.0, 0.0, 0.0)], spacegroup=141,
                   cellpar=[4.81, 4.81, 2.65, 90, 90, 90])


def gamma_b28() -> Atoms:
    """gamma-B28, Pnnm (58), 28 atoms, Oganov et al. Nature 2009 (0 GPa)."""
    return crystal(
        symbols=['B'] * 5,
        basis=[
            (0.1735, 0.5109, 0.0),
            (0.0876, 0.2210, 0.0),
            (0.3223, 0.2293, 0.0),
            (0.1649, 0.3685, 0.1893),
            (0.3343, 0.0773, 0.1869),
        ],
        spacegroup=58,
        cellpar=[5.0576, 5.6245, 6.9884, 90, 90, 90],
    )


def quartz_alpha() -> Atoms:
    """alpha-quartz SiO2, P3_121 (152), 9 atoms.

    Si on Wyckoff 3a requires z=1/3 in the ASE setting of #152; the O
    z-coordinate is shifted by +1/3 accordingly (Si-O = 1.60 A).
    """
    return crystal(['Si', 'O'],
                   basis=[(0.4697, 0.0, 1.0 / 3.0),
                          (0.4135, 0.2669, 0.4524)],
                   spacegroup=152,
                   cellpar=[4.913, 4.913, 5.405, 90, 90, 120])


def cristobalite_alpha() -> Atoms:
    """alpha-cristobalite SiO2, P4_12_12 (92), 12 atoms."""
    return crystal(['Si', 'O'],
                   basis=[(0.3047, 0.3047, 0.0), (0.2381, 0.1109, 0.1826)],
                   spacegroup=92,
                   cellpar=[4.9709, 4.9709, 6.9278, 90, 90, 90])


def tio2_rutile() -> Atoms:
    """Rutile TiO2, P4_2/mnm (136), 6 atoms."""
    return crystal(['Ti', 'O'],
                   basis=[(0.0, 0.0, 0.0), (0.3053, 0.3053, 0.0)],
                   spacegroup=136,
                   cellpar=[4.5937, 4.5937, 2.9587, 90, 90, 90])


def tio2_anatase() -> Atoms:
    """Anatase TiO2, I4_1/amd (141), 12 atoms (conv)."""
    return crystal(['Ti', 'O'],
                   basis=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.2081)],
                   spacegroup=141,
                   cellpar=[3.7845, 3.7845, 9.5143, 90, 90, 90])


def tio2_brookite() -> Atoms:
    """Brookite TiO2, Pbca (61), 24 atoms (Meagher & Lager 1979)."""
    return crystal(['Ti', 'O', 'O'],
                   basis=[(0.1289, 0.0972, 0.8628),
                          (0.0095, 0.1491, 0.1835),
                          (0.2314, 0.1110, 0.5366)],
                   spacegroup=61,
                   cellpar=[9.174, 5.449, 5.138, 90, 90, 90])


def spinel_mgal2o4() -> Atoms:
    """MgAl2O4 spinel, Fd-3m (227), primitive 14-atom cell.

    Built as the 56-atom conventional cell then reduced to primitive
    via spglib so the atom count matches the search composition.
    """
    conv = crystal(['Mg', 'Al', 'O'],
                   basis=[(0.125, 0.125, 0.125),
                          (0.5, 0.5, 0.5),
                          (0.2624, 0.2624, 0.2624)],
                   spacegroup=227,
                   cellpar=[8.0832, 8.0832, 8.0832, 90, 90, 90],
                   setting=2)
    return to_primitive(conv)


def mgsio3_perovskite() -> Atoms:
    """MgSiO3 perovskite (bridgmanite), Pbnm (62), 20 atoms.

    Horiuchi et al. 1987 coordinates, expressed in the standard Pnma
    setting via the axis permutation (x,y,z)_Pbnm -> (y,z,x)_Pnma.
    """
    # Pbnm: a=4.7787, b=4.9313, c=6.9083
    # Mg (0.5131, 0.5562, 1/4); Si (0.5, 0.0, 0.5);
    # O1 (0.1023, 0.4664, 1/4); O2 (0.1961, 0.2014, 0.5531)
    def pbnm_to_pnma(x, y, z):
        return (y, z, x)
    return crystal(['Mg', 'Si', 'O', 'O'],
                   basis=[pbnm_to_pnma(0.5131, 0.5562, 0.25),
                          pbnm_to_pnma(0.5, 0.0, 0.5),
                          pbnm_to_pnma(0.1023, 0.4664, 0.25),
                          pbnm_to_pnma(0.1961, 0.2014, 0.5531)],
                   spacegroup=62,
                   cellpar=[4.9313, 6.9083, 4.7787, 90, 90, 90])


def mgsio3_ilmenite() -> Atoms:
    """MgSiO3 akimotoite (ilmenite-type), R-3 (148), 30-atom hex cell.

    Horiuchi et al. 1982. Calibration competitor for perovskite.
    """
    return crystal(['Mg', 'Si', 'O'],
                   basis=[(0.0, 0.0, 0.3599),
                          (0.0, 0.0, 0.1581),
                          (0.3203, 0.0354, 0.2408)],
                   spacegroup=148,
                   cellpar=[4.7284, 4.7284, 13.5591, 90, 90, 120])


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def to_primitive(atoms: Atoms, symprec: float = 1e-3) -> Atoms:
    """Reduce to the spglib primitive cell."""
    import spglib
    cell = (atoms.cell.array, atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    lattice, scaled, numbers = spglib.find_primitive(cell, symprec=symprec)
    return Atoms(numbers=numbers, scaled_positions=scaled, cell=lattice,
                 pbc=True)


def composition_of(atoms: Atoms) -> Dict[str, int]:
    symbols = atoms.get_chemical_symbols()
    return {s: symbols.count(s) for s in sorted(set(symbols))}


def ref_at_composition(spec: SystemSpec, ref: Atoms,
                       max_factor: int = 4) -> Optional[Atoms]:
    """Return a diagonal supercell of *ref* matching spec.composition.

    Returns None if no diagonal supercell matches (then only the
    enthalpy + spacegroup success criterion applies for this ref).
    """
    ref_comp = composition_of(ref)
    if set(ref_comp) != set(spec.composition):
        return None
    ratios = {s: spec.composition[s] / ref_comp[s] for s in ref_comp}
    ratio = next(iter(ratios.values()))
    if any(abs(r - ratio) > 1e-9 for r in ratios.values()):
        return None
    if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
        return None
    target_mult = int(round(ratio))
    for na in range(1, max_factor + 1):
        for nb in range(1, max_factor + 1):
            for nc in range(1, max_factor + 1):
                if na * nb * nc == target_mult:
                    return ref.repeat((na, nb, nc))
    return None


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

SYSTEMS: Dict[str, SystemSpec] = {
    'si16': SystemSpec(
        name='si16', composition={'Si': 16}, pressure_GPa=0.0,
        difficulty='easy',
        refs={'diamond': si_diamond, 'beta_tin': si_beta_tin},
        expected_gs='diamond',
        vol_per_atom_range=(14.0, 26.0), min_dist_ang=1.8),
    'b28': SystemSpec(
        name='b28', composition={'B': 28}, pressure_GPa=50.0,
        difficulty='easy',
        refs={'gamma_b28': gamma_b28},
        expected_gs='gamma_b28',
        vol_per_atom_range=(4.0, 10.0), min_dist_ang=1.3),
    'tio2_24': SystemSpec(
        name='tio2_24', composition={'Ti': 8, 'O': 16}, pressure_GPa=0.0,
        difficulty='medium',
        refs={'rutile': tio2_rutile, 'anatase': tio2_anatase,
              'brookite': tio2_brookite},
        expected_gs='rutile',
        vol_per_atom_range=(8.0, 16.0), min_dist_ang=1.6),
    'spinel': SystemSpec(
        name='spinel', composition={'Mg': 2, 'Al': 4, 'O': 8},
        pressure_GPa=0.0, difficulty='medium',
        refs={'spinel': spinel_mgal2o4},
        expected_gs='spinel',
        vol_per_atom_range=(6.5, 13.0), min_dist_ang=1.6),
    # NOTE: composition must tile the potential's GS cell. On MatterSim
    # the SiO2 GS is alpha-cristobalite (Z=4, 12 atoms) and quartz is
    # unstable (collapses to C2), so the cell is 24 atoms (2 cristobalite
    # cells), not 18 (which only tiles quartz).
    'sio2_24': SystemSpec(
        name='sio2_24', composition={'Si': 8, 'O': 16}, pressure_GPa=0.0,
        difficulty='hard',
        refs={'quartz': quartz_alpha, 'cristobalite': cristobalite_alpha},
        expected_gs='cristobalite',
        vol_per_atom_range=(9.0, 20.0), min_dist_ang=1.4),
    'mgsio3_20': SystemSpec(
        name='mgsio3_20', composition={'Mg': 4, 'Si': 4, 'O': 12},
        pressure_GPa=30.0, difficulty='hard',
        refs={'perovskite': mgsio3_perovskite,
              'ilmenite': mgsio3_ilmenite},
        expected_gs='perovskite',
        vol_per_atom_range=(5.0, 11.0), min_dist_ang=1.4),
}
