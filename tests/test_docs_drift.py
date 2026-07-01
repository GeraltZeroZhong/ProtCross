from pathlib import Path

from protcross.assets import DEFAULT_ASSET_BUNDLE
from protcross.cli.main import COMMANDS
from protcross.cli.predict import _default_output_paths


def test_readme_documents_current_cli_commands_and_default_outputs():
    readme = Path("README.md").read_text(encoding="utf-8")

    for command in COMMANDS:
        assert f"protcross {command}" in readme

    output_paths = _default_output_paths("input.pdb")
    for path in output_paths.values():
        assert path.name in readme


def test_readme_asset_names_match_default_bundle():
    readme = Path("README.md").read_text(encoding="utf-8")

    for spec in DEFAULT_ASSET_BUNDLE.assets:
        assert spec.filename in readme


def test_readme_pypi_predict_example_acknowledges_esm_license_gate():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "protcross predict input.pdb --accept-esm-license --output input.protcross.pdb" in readme


def test_legacy_readme_modern_predict_example_acknowledges_esm_license_gate():
    legacy_readme = Path("reproduction/legacy/README.md").read_text(encoding="utf-8")

    assert "protcross predict input.pdb --accept-esm-license" in legacy_readme


def test_examples_do_not_bypass_hash_checks_for_release_assets():
    examples_readme = Path("examples/README.md").read_text(encoding="utf-8")
    release_section = examples_readme.split("For source checkout testing", 1)[1]

    assert "--trust-unverified-assets" not in release_section.split("Use `--trust-unverified-assets`", 1)[0]
