from pathlib import Path

import pytest


PUBLISH_ROOT = Path("/home/zero/ProtCross-all/ProtCross-publish")
PUBLISH_PDB_SAMPLE = PUBLISH_ROOT / "data/processed_pdb/6fhu.pt"
PUBLISH_RAW_PDB = PUBLISH_ROOT / "data/raw_pdb/6fhu.pdb"
PUBLISH_ESM_WEIGHTS = PUBLISH_ROOT / "esmc_weights/esmc_600m_2024_12_v0.pth"


def require_file(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"test fixture not found: {path}")
    return path

