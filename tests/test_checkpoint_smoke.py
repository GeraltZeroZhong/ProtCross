import torch
from pathlib import Path
from torch_geometric.data import Batch, Data

from evopoint_da.models import EvoPointDALitModule

from conftest import PUBLISH_PDB_SAMPLE, require_file


def test_checkpoint_forward_cpu_on_publish_sample():
    sample_path = require_file(PUBLISH_PDB_SAMPLE)
    checkpoint = require_file(Path("checkpoint/best-epoch=59.ckpt"))
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
