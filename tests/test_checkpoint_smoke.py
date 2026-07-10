import os
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from protcross.models import EvoPointDALitModule

from conftest import PUBLISH_PDB_SAMPLE, require_file


DEFAULT_RELEASE_CHECKPOINT = Path("checkpoints/protcross-0.1.2-binding-moad-final.ckpt")


def test_checkpoint_forward_cpu_on_publish_sample():
    checkpoint = _release_checkpoint()
    raw = _load_release_sample()
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
        scores = torch.softmax(logits, dim=1)[:, 1]

    assert logits.shape == (batch.x.shape[0], 2)
    assert torch.isfinite(scores).all()
    assert torch.all((scores >= 0) & (scores <= 1))
    if os.environ.get("PROTCROSS_RELEASE_SYNTHETIC_SAMPLE") == "1":
        expected_feature_prefix = torch.tensor(
            [
                -42.04204,
                -31.53744,
                155.18114,
                0.30038,
                27.11786,
                -16.36558,
                -41.44625,
                -7.21677,
                38.80212,
                12.01224,
            ]
        )
        torch.testing.assert_close(feats[0, :10], expected_feature_prefix, rtol=2e-3, atol=5e-2)
        assert float(feats.mean()) == pytest.approx(1.78408, abs=5e-2)
    else:
        expected_prefix = torch.tensor(
            [
                0.24724323,
                0.33811420,
                0.20718010,
                0.50588417,
                0.47987068,
                0.18985108,
                0.55295146,
                0.32932904,
                0.36394376,
                0.34797800,
            ]
        )
        torch.testing.assert_close(scores[:10], expected_prefix, rtol=5e-4, atol=5e-4)
        assert float(scores.mean()) == pytest.approx(0.30683029, abs=5e-4)
        assert torch.nonzero(scores > 0.5, as_tuple=False).flatten().tolist() == [3, 6, 41]


def _load_release_sample() -> dict[str, torch.Tensor]:
    if os.environ.get("PROTCROSS_RELEASE_SYNTHETIC_SAMPLE") == "1":
        generator = torch.Generator().manual_seed(13)
        node_count = int(os.environ.get("PROTCROSS_RELEASE_SYNTHETIC_NODES", "128"))
        return {
            "x": torch.randn(node_count, 128, generator=generator),
            "pos": torch.randn(node_count, 3, generator=generator) * 0.1,
            "plddt": torch.full((node_count,), 90.0),
            "y": torch.zeros(node_count, dtype=torch.long),
        }
    sample_path = require_file(PUBLISH_PDB_SAMPLE)
    return torch.load(sample_path, map_location="cpu", weights_only=False)


def _release_checkpoint() -> Path:
    checkpoint = Path(os.environ.get("PROTCROSS_RELEASE_CHECKPOINT", DEFAULT_RELEASE_CHECKPOINT))
    if checkpoint.exists():
        return checkpoint
    message = f"release checkpoint fixture not found: {checkpoint}"
    if os.environ.get("PROTCROSS_RELEASE_SMOKE") == "1":
        pytest.fail(message)
    pytest.skip(message)
