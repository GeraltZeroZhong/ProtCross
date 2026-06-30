import os
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from protcross.models import EvoPointDALitModule

from conftest import PUBLISH_PDB_SAMPLE, require_file


DEFAULT_RELEASE_CHECKPOINT = Path("checkpoints/protcross-0.1.2-binding-moad-final.ckpt")


def test_checkpoint_forward_cpu_on_publish_sample():
    sample_path = require_file(PUBLISH_PDB_SAMPLE)
    checkpoint = _release_checkpoint()
    raw = torch.load(sample_path, map_location="cpu", weights_only=False)
    batch = Batch.from_data_list(
        [
            Data(
                x=raw["x"].float(),
                pos=raw["pos"].float(),
                plddt=raw["plddt"].float(),
                y=raw["y"].long(),
            )
        ]
    )

    model = EvoPointDALitModule.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model.eval()
    with torch.no_grad():
        feats, _ = model.backbone(batch.x, batch.pos, batch.batch)
        logits = model.seg_head(feats)

    assert logits.shape == (batch.x.shape[0], 2)


def _release_checkpoint() -> Path:
    checkpoint = Path(os.environ.get("PROTCROSS_RELEASE_CHECKPOINT", DEFAULT_RELEASE_CHECKPOINT))
    if checkpoint.exists():
        return checkpoint
    message = f"release checkpoint fixture not found: {checkpoint}"
    if os.environ.get("PROTCROSS_RELEASE_SMOKE") == "1":
        pytest.fail(message)
    pytest.skip(message)
