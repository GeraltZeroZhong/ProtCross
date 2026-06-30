from __future__ import annotations

import sys
import warnings
import importlib
from importlib import resources
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class


def test_gitignore_does_not_hide_src_protcross_data():
    lines = Path(".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/data/" in lines
    assert "data/" not in lines


def test_hydra_data_target_uses_protcross_package():
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
        cfg = compose(config_name="train")

    assert cfg.data._target_ == "protcross.data.datamodule.EvoPointDataModule"
    cls = get_class(cfg.data._target_)
    assert cls.__name__ == "EvoPointDataModule"


def test_packaged_hydra_configs_are_available():
    config_root = resources.files("protcross.configs")

    assert (config_root / "train.yaml").is_file()
    assert (config_root / "data" / "protein_seg.yaml").is_file()


def test_root_and_packaged_hydra_configs_stay_in_sync():
    for relative_path in (
        "train.yaml",
        "data/protein_seg.yaml",
        "model/da_module.yaml",
        "trainer/default.yaml",
    ):
        root_config = Path("configs") / relative_path
        packaged_config = Path("src/protcross/configs") / relative_path
        assert packaged_config.read_text(encoding="utf-8") == root_config.read_text(encoding="utf-8")


def test_train_help_path_does_not_enter_training(capsys):
    from protcross.cli.train import main

    assert main(["--help"], prog="protcross train") == 0
    help_text = capsys.readouterr().out
    assert "Run the ProtCross Hydra training workflow." in help_text
    assert "trainer.max_epochs=10" in help_text


def test_evopoint_da_inference_alias_emits_deprecation_warning():
    for name in list(sys.modules):
        if name == "evopoint_da" or name.startswith("evopoint_da."):
            del sys.modules[name]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        from evopoint_da.inference import PredictionResult

    assert PredictionResult.__module__.startswith("protcross.")
    assert any("renamed to protcross" in str(item.message) for item in caught)


def test_evopoint_da_alias_import_matrix():
    aliases = (
        "evopoint_da.assets",
        "evopoint_da.cli.predict",
        "evopoint_da.data.components",
        "evopoint_da.data.dataset",
        "evopoint_da.models.module",
        "evopoint_da.models.backbones.pointnet2",
    )
    for name in list(sys.modules):
        if name == "evopoint_da" or name.startswith("evopoint_da."):
            del sys.modules[name]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for alias in aliases:
            module = importlib.import_module(alias)
            assert module.__name__.startswith("protcross.")


def test_readme_does_not_reference_old_package_or_unarchived_legacy_paths():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "evopoint_da" not in readme
    assert "python train.py" not in readme
    assert "scripts/preprocess_esm.py" not in readme
    assert "0.1.0" not in readme
    assert "0.1.1" not in readme
    assert "0.1.1-paper" not in readme
    assert "PDBbind" not in readme
    assert "best-epoch=59.ckpt" not in readme
    assert "protcross-predict" not in readme
    assert "run_Predict_ProtCross.py" not in readme
    assert "python reproduction/legacy/" not in readme


def test_legacy_readme_owns_original_reproducibility_details():
    readme = Path("reproduction/legacy/README.md").read_text(encoding="utf-8")

    assert "0.1.0" in readme
    assert "0.1.1-paper" in readme
    assert "PDBbind" in readme
    assert "best-epoch=59.ckpt" in readme
    assert "python reproduction/legacy/run_Predict_ProtCross.py" in readme
