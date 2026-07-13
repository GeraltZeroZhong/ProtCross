import re
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


def test_readme_prediction_examples_acknowledge_esm_license_gate():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "protcross setup-assets --accept-esm-license" in readme
    existing_assets = readme.split("### Existing or custom assets", 1)[1]
    existing_esm_example = existing_assets.split("Explicit release assets", 1)[0]
    assert "--esm-weights /absolute/path/to/esmc_600m_2024_12_v0.pth" in existing_esm_example
    assert "--accept-esm-license" in existing_esm_example


def test_repository_has_one_project_markdown_document():
    ignored_parts = {".pytest_cache", "node_modules", "target"}
    markdown_files = {
        path
        for path in Path(".").rglob("*.md")
        if not ignored_parts.intersection(path.parts)
    }

    assert markdown_files == {Path("README.md")}


def test_readme_does_not_reference_removed_markdown_documents():
    readme = Path("README.md").read_text(encoding="utf-8")

    for removed in (
        "MODEL_CARD.md",
        "RELEASE_NOTES_0.1.2.md",
        "RELEASE_NOTES_0.2.1.md",
        "desktop/README.md",
        "examples/README.md",
        "reproduction/legacy/README.md",
    ):
        assert removed not in readme


def test_readme_explains_custom_asset_verification():
    readme = Path("README.md").read_text(encoding="utf-8")
    asset_section = readme.split("### Existing or custom assets", 1)[1].split(
        "## Desktop application", 1
    )[0]

    assert "verified against the selected bundle's SHA256" in asset_section
    assert "--trust-unverified-assets" in asset_section
    assert "real hashes and verification status" in asset_section


def test_readme_front_matter_is_direct_and_ordered_for_first_use():
    readme = Path("README.md").read_text(encoding="utf-8")
    intro = readme.split("## Quick start", 1)[0]
    compact_intro = " ".join(intro.split())
    front_matter = readme.split("## Contents", 1)[0]

    assert "protein binding-site prediction tool" in intro
    assert "ProtCross: Bridging the PDB-AlphaFold Gap for Binding Site Prediction" in compact_intro
    assert "Journal of Chemical Information and Modeling" in intro
    assert "https://doi.org/10.1021/acs.jcim.5c03224" in intro
    assert readme.index("## Quick start") < readme.index("## Contents")
    assert readme.index("## Contents") < readme.index("## Highlights")
    assert "### Score and threshold" not in front_matter
    assert "| Component |" not in front_matter
    assert "Version note" not in front_matter


def test_readme_front_matter_exposes_install_surfaces_and_project_significance():
    readme = Path("README.md").read_text(encoding="utf-8")
    intro = readme.split("## Quick start", 1)[0]

    assert "https://pypi.org/project/protcross/" in intro
    assert "Windows x64 Desktop" in intro
    assert "macOS Apple Silicon Desktop" in intro
    assert "Global fold accuracy and pLDDT alone do not establish" in intro
    assert "residue-level binding-site scores" in intro


def test_readme_022_history_has_at_most_three_entries():
    readme = Path("README.md").read_text(encoding="utf-8")
    release = readme.split("### 0.2.2", 1)[1].split("### 0.2.1", 1)[0]
    entries = [line for line in release.splitlines() if line.startswith("- ")]

    assert 1 <= len(entries) <= 3


def test_readme_uses_technical_structure_without_review_defense_sections():
    readme = Path("README.md").read_text(encoding="utf-8")
    lowered = readme.lower()

    for heading in (
        "## Model and inference pipeline",
        "## Batch inference",
        "## Output package",
        "## Training and development",
        "## Version history",
        "## Citation",
    ):
        assert heading in readme
    assert "scientific scope and limitations" not in lowered
    assert "interpretation boundaries" not in lowered
    assert "rather than" not in lowered
    assert "instead of" not in lowered
    assert re.search(r"\bnot\b.{0,80}\bbut\b", lowered) is None


def test_readme_contains_no_emoji():
    readme = Path("README.md").read_text(encoding="utf-8")
    emoji = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF]")

    assert emoji.search(readme) is None
    assert "\ufe0f" not in readme
