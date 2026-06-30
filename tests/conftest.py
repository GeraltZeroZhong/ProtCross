from pathlib import Path
import socket

import pytest
import requests


PUBLISH_ROOT = Path("/home/zero/ProtCross-all/ProtCross-publish")
PUBLISH_PDB_SAMPLE = PUBLISH_ROOT / "data/processed_pdb/6fhu.pt"
PUBLISH_RAW_PDB = PUBLISH_ROOT / "data/raw_pdb/6fhu.pdb"
PUBLISH_ESM_WEIGHTS = PUBLISH_ROOT / "esmc_weights/esmc_600m_2024_12_v0.pth"


def require_file(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"test fixture not found: {path}")
    return path


@pytest.fixture(autouse=True)
def block_network_requests(monkeypatch, request):
    if request.node.get_closest_marker("network"):
        return

    def blocked_request(*args, **kwargs):
        raise RuntimeError("Network access is disabled during tests.")

    def blocked_connect(*args, **kwargs):
        raise RuntimeError("Network access is disabled during tests.")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked_request)
    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
