"""Download and validate ProtCross runtime assets."""

from __future__ import annotations

import hashlib
import os
import shlex
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm


RELEASE_TAG = "v0.1.2"
GITHUB_RELEASE_BASE = f"https://github.com/GeraltZeroZhong/ProtCross/releases/download/{RELEASE_TAG}"
DEFAULT_ASSETS_DIR = Path.home() / ".cache" / "protcross" / "assets" / RELEASE_TAG

DEFAULT_ESM_URL = (
    "https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12/"
    "resolve/main/data/weights/esmc_600m_2024_12_v0.pth"
)
DEFAULT_CHECKPOINT_FILENAME = "protcross-0.1.2-binding-moad-final.ckpt"
DEFAULT_PCA_FILENAME = "pca_esmc_128_binding_moad_0.1.2.pkl"
LEGACY_CHECKPOINT_FILENAME = "best-epoch=59.ckpt"
LEGACY_PCA_FILENAME = "pca_esmc_128.pkl"
DEFAULT_CHECKPOINT_URL = f"{GITHUB_RELEASE_BASE}/{DEFAULT_CHECKPOINT_FILENAME}"
DEFAULT_PCA_URL = f"{GITHUB_RELEASE_BASE}/{DEFAULT_PCA_FILENAME}"


@dataclass(frozen=True)
class AssetSpec:
    name: str
    filename: str
    url: str
    sha256: str | None = None


DEFAULT_ASSETS = (
    AssetSpec(
        name="ESM-C 600M weights",
        filename="esmc_600m_2024_12_v0.pth",
        url=DEFAULT_ESM_URL,
        sha256="8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324",
    ),
    AssetSpec(
        name="ProtCross checkpoint",
        filename=DEFAULT_CHECKPOINT_FILENAME,
        url=DEFAULT_CHECKPOINT_URL,
        sha256="ccb56884b21402a027bfae9d4779f38c8f534513d980a96d7cd78c9931748b65",
    ),
    AssetSpec(
        name="ProtCross PCA reducer",
        filename=DEFAULT_PCA_FILENAME,
        url=DEFAULT_PCA_URL,
        sha256="0f4e11806a622642c07dad539cec4216030220c1b5f3fc44c7926a2f6bca4d62",
    ),
)


def get_default_assets_dir() -> Path:
    return Path(os.environ.get("PROTCROSS_ASSETS_DIR", DEFAULT_ASSETS_DIR)).expanduser()


def setup_assets(
    output_dir: str | Path | None = None,
    *,
    esm_url: str = DEFAULT_ESM_URL,
    checkpoint_url: str = DEFAULT_CHECKPOINT_URL,
    pca_url: str = DEFAULT_PCA_URL,
    force: bool = False,
    verify: bool = True,
    skip_esm: bool = False,
) -> Path:
    """Download ProtCross assets and return the asset directory."""
    output_dir = Path(output_dir).expanduser() if output_dir else get_default_assets_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        AssetSpec(
            DEFAULT_ASSETS[0].name,
            DEFAULT_ASSETS[0].filename,
            esm_url,
            DEFAULT_ASSETS[0].sha256,
        ),
        AssetSpec(
            DEFAULT_ASSETS[1].name,
            DEFAULT_ASSETS[1].filename,
            checkpoint_url,
            DEFAULT_ASSETS[1].sha256,
        ),
        AssetSpec(
            DEFAULT_ASSETS[2].name,
            DEFAULT_ASSETS[2].filename,
            pca_url,
            DEFAULT_ASSETS[2].sha256,
        ),
    ]
    if skip_esm:
        specs = specs[1:]

    print(f"Installing ProtCross assets into {output_dir}")
    print("Note: ESM-C weights are distributed by EvolutionaryScale under their Hugging Face model terms.")
    for spec in specs:
        download_asset(spec, output_dir / spec.filename, force=force, verify=verify)

    write_env_file(output_dir, include_esm=(output_dir / DEFAULT_ASSETS[0].filename).exists())
    print("\nAsset setup complete.")
    print("Use with: protcross predict input.pdb --output output.pdb")
    print(f"Environment file written to: {output_dir / 'protcross.env'}")
    return output_dir


def download_asset(spec: AssetSpec, output_path: Path, *, force: bool = False, verify: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        if not verify or not spec.sha256 or sha256_file(output_path) == spec.sha256:
            print(f"[skip] {spec.name}: {output_path}")
            return
        print(f"[warn] Existing file failed SHA256 verification and will be replaced: {output_path}")

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[download] {spec.name}")
    print(f"           {spec.url}")
    with requests.get(spec.url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with tmp_path.open("wb") as file, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=spec.filename,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                progress.update(len(chunk))

    if verify and spec.sha256:
        actual = sha256_file(tmp_path)
        if actual != spec.sha256:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch for {spec.filename}: expected {spec.sha256}, got {actual}"
            )

    tmp_path.replace(output_path)
    print(f"[ok] {output_path}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_env_file(output_dir: Path, *, include_esm: bool = True) -> None:
    env_path = output_dir / "protcross.env"
    lines = [
        _export_line("PROTCROSS_ASSETS_DIR", output_dir),
        _export_line("PROTCROSS_CHECKPOINT", output_dir / DEFAULT_CHECKPOINT_FILENAME),
    ]
    if include_esm:
        lines.append(_export_line("PROTCROSS_ESM_WEIGHTS", output_dir / "esmc_600m_2024_12_v0.pth"))
    lines.extend(
        [
            _export_line("PROTCROSS_PCA", output_dir / DEFAULT_PCA_FILENAME),
            "",
        ]
    )
    env_path.write_text("\n".join(lines), encoding="utf-8")


def _export_line(name: str, value: str | Path) -> str:
    return f"export {name}={shlex.quote(str(value))}"
