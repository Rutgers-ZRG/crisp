"""Template for configuring CRISP with a Slurm-backed relaxation backend.

Set paths through environment variables or replace this file with your own
configuration layer. This example intentionally avoids private cluster paths.
"""

import os

from crisp import CRISPSearch, FingerprintCalculator, HPCRelaxer, RelaxBackend
from crisp.hpc import make_subprocess_adapter


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this example")
    return value


def main() -> None:
    ssh_host = require_env("CRISP_SSH_HOST")
    remote_base = require_env("CRISP_REMOTE_BASE")
    model_path = os.environ.get("CRISP_MODEL_PATH", "")
    conda_env = os.environ.get("CRISP_CONDA_ENV", "crisp")
    conda_path = os.environ.get("CRISP_CONDA_PATH", "conda")

    ssh_exec, upload, download = make_subprocess_adapter(ssh_host)

    hpc = HPCRelaxer(
        ssh_exec=ssh_exec,
        upload=upload,
        download=download,
        remote_base=remote_base,
        backend=RelaxBackend.MACE_MP,
        model_path=model_path,
        conda_env=conda_env,
        conda_path=conda_path,
        partition=os.environ.get("CRISP_SLURM_PARTITION", "gpu"),
        gres=os.environ.get("CRISP_SLURM_GRES", "gpu:1"),
        time_limit=os.environ.get("CRISP_SLURM_TIME", "00:30:00"),
        mem=os.environ.get("CRISP_SLURM_MEM", "16GB"),
        device=os.environ.get("CRISP_DEVICE", "cuda"),
    )

    fp_calc = FingerprintCalculator(cutoff=6.0, natx=100, orbital="s")
    search = CRISPSearch(
        hpc_relaxer=hpc,
        fp_calc=fp_calc,
        composition={"Si": 8},
        n_random=20,
        n_select=8,
        max_generations=3,
        checkpoint_dir=os.environ.get("CRISP_CHECKPOINT_DIR"),
    )
    archive = search.run()
    print(f"Found {len(archive.entries)} unique relaxed structures")


if __name__ == "__main__":
    main()
