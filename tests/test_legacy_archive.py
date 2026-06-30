from pathlib import Path
import py_compile
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = REPO_ROOT / "reproduction" / "legacy"


def test_legacy_archive_contains_reproducibility_scripts():
    expected = {
        "run_Predict_ProtCross.py",
        "train.py",
        "preprocess_esm.py",
        "get_af2.py",
        "map_labels.py",
        "map_labels-o.py",
        "setup_assets.py",
        "run_Strategy.py",
        "run_multiseed_benchmark.py",
        "test_adaptive.py",
        "eval_run.py",
        "eval_dataset.py",
        "eval_utils.py",
        "analyze_geometric.py",
        "sensitivity-cutoff.py",
    }

    assert {path.name for path in LEGACY_DIR.glob("*.py")} == expected
    assert (LEGACY_DIR / "README.md").exists()
    assert (LEGACY_DIR / "pdb_uniprot_mapping.json").exists()


def test_old_top_level_legacy_entrypoints_are_not_exposed():
    for relative_path in (
        "run_Predict_ProtCross.py",
        "train.py",
        "run_Strategy.py",
        "run_multiseed_benchmark.py",
        "test_adaptive.py",
        "scripts/preprocess_esm.py",
        "scripts/get_af2.py",
        "scripts/map_labels.py",
        "scripts/setup_assets.py",
        "pdb_uniprot_mapping.json",
    ):
        assert not (REPO_ROOT / relative_path).exists()


def test_legacy_archive_scripts_compile():
    for path in LEGACY_DIR.glob("*.py"):
        py_compile.compile(str(path), doraise=True)


def test_legacy_wrappers_help_smoke():
    for script_name in (
        "run_Predict_ProtCross.py",
        "preprocess_esm.py",
        "get_af2.py",
        "map_labels.py",
        "setup_assets.py",
        "train.py",
        "test_adaptive.py",
    ):
        result = subprocess.run(
            [sys.executable, "-B", str(LEGACY_DIR / script_name), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_legacy_wrappers_point_to_current_package_layout():
    wrapper_text = (LEGACY_DIR / "run_Predict_ProtCross.py").read_text(encoding="utf-8")
    assert "../../src" in wrapper_text
    assert "protcross.cli.predict" in wrapper_text

    train_text = (LEGACY_DIR / "train.py").read_text(encoding="utf-8")
    assert "../../configs" in train_text
    assert "protcross.training" in train_text

    map_labels_text = (LEGACY_DIR / "map_labels-o.py").read_text(encoding="utf-8")
    assert "reproduction/legacy/map_labels.py" in map_labels_text
