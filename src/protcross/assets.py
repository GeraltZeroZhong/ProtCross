"""Download and validate ProtCross runtime assets."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


RELEASE_TAG = "v0.1.2"
DEFAULT_ASSET_VERSION = "0.1.2"
LEGACY_ASSET_VERSION = "0.1.1-paper"
ASSET_MANIFEST_FILENAME = "protcross-assets.json"
GITHUB_RELEASE_BASE = f"https://github.com/GeraltZeroZhong/ProtCross/releases/download/{RELEASE_TAG}"
DEFAULT_ASSETS_DIR = Path.home() / ".cache" / "protcross" / "assets" / RELEASE_TAG

DEFAULT_ESM_URL = (
    "https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12/"
    "resolve/main/data/weights/esmc_600m_2024_12_v0.pth"
)
DEFAULT_CHECKPOINT_FILENAME = "protcross-0.1.2-binding-moad-final.ckpt"
DEFAULT_PCA_FILENAME = "pca_esmc_128_binding_moad_0.1.2.pkl"
LEGACY_CHECKPOINT_FILENAME = "best-epoch=59.ckpt"
LEGACY_CHECKPOINT_RELEASE_FILENAME = "best-epoch.59.ckpt"
LEGACY_PCA_FILENAME = "pca_esmc_128.pkl"
DEFAULT_CHECKPOINT_URL = f"{GITHUB_RELEASE_BASE}/{DEFAULT_CHECKPOINT_FILENAME}"
DEFAULT_PCA_URL = f"{GITHUB_RELEASE_BASE}/{DEFAULT_PCA_FILENAME}"
ESM_LICENSE_URL = "https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement"
ESM_LICENSE_ACCEPT_ENV = "PROTCROSS_ACCEPT_ESM_LICENSE"
TRUST_UNVERIFIED_ASSETS_ENV = "PROTCROSS_TRUST_UNVERIFIED_ASSETS"
LEGACY_RELEASE_TAG = "v0.1.1"
LEGACY_GITHUB_RELEASE_BASE = f"https://github.com/GeraltZeroZhong/ProtCross/releases/download/{LEGACY_RELEASE_TAG}"
LEGACY_CHECKPOINT_URL = f"{LEGACY_GITHUB_RELEASE_BASE}/{LEGACY_CHECKPOINT_RELEASE_FILENAME}"
LEGACY_PCA_URL = f"{LEGACY_GITHUB_RELEASE_BASE}/{LEGACY_PCA_FILENAME}"


@dataclass(frozen=True)
class AssetSpec:
    name: str
    filename: str
    url: str
    sha256: str | None = None


@dataclass(frozen=True)
class AssetBundle:
    version: str
    release_tag: str
    checkpoint_filename: str
    pca_filename: str
    esm_filename: str
    pca_dim: int
    esm_model: str
    training_description: str
    assets: tuple[AssetSpec, ...]


@dataclass(frozen=True)
class PredictorAssets:
    """Resolved paths for the three runtime files needed by prediction.

    This class intentionally lives in the light-weight assets module so CLI
    help and asset resolution do not import torch or the model stack.
    """

    checkpoint: Path
    esm_weights: Path
    pca: Path
    asset_version: str = DEFAULT_ASSET_VERSION

    @classmethod
    def from_dir(cls, assets_dir: str | Path, *, asset_version: str = "default") -> "PredictorAssets":
        assets_dir = Path(assets_dir).expanduser()
        manifest = read_asset_manifest(assets_dir)
        if manifest:
            bundle = get_asset_bundle(str(manifest.get("asset_version", asset_version)))
        else:
            bundle = _infer_manifestless_bundle(assets_dir, asset_version)
        return cls(
            checkpoint=assets_dir / bundle.checkpoint_filename,
            esm_weights=assets_dir / bundle.esm_filename,
            pca=assets_dir / bundle.pca_filename,
            asset_version=bundle.version,
        )

    @classmethod
    def from_default_dir(cls, *, asset_version: str = "default") -> "PredictorAssets":
        return cls.from_dir(get_default_assets_dir(asset_version), asset_version=asset_version)

    @classmethod
    def from_env(cls) -> "PredictorAssets":
        missing = [
            name
            for name in ("PROTCROSS_CHECKPOINT", "PROTCROSS_ESM_WEIGHTS", "PROTCROSS_PCA")
            if not os.environ.get(name)
        ]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        return cls(
            checkpoint=Path(os.environ["PROTCROSS_CHECKPOINT"]).expanduser(),
            esm_weights=Path(os.environ["PROTCROSS_ESM_WEIGHTS"]).expanduser(),
            pca=Path(os.environ["PROTCROSS_PCA"]).expanduser(),
            asset_version=os.environ.get("PROTCROSS_ASSET_VERSION", "env"),
        )

    def missing_files(self) -> list[Path]:
        return [path for path in (self.checkpoint, self.esm_weights, self.pca) if not path.exists()]

    def is_complete(self) -> bool:
        return not self.missing_files()


@dataclass(frozen=True)
class ResolvedPredictionAssets:
    checkpoint: Path
    esm_weights: Path
    pca: Path
    assets: PredictorAssets | None
    asset_version: str


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

DEFAULT_ASSET_BUNDLE = AssetBundle(
    version=DEFAULT_ASSET_VERSION,
    release_tag=RELEASE_TAG,
    checkpoint_filename=DEFAULT_CHECKPOINT_FILENAME,
    pca_filename=DEFAULT_PCA_FILENAME,
    esm_filename=DEFAULT_ASSETS[0].filename,
    pca_dim=128,
    esm_model="esmc-600m-2024-12",
    training_description="Binding MOAD-derived release checkpoint for practical prediction and benchmark use.",
    assets=DEFAULT_ASSETS,
)

LEGACY_ASSET_BUNDLE = AssetBundle(
    version=LEGACY_ASSET_VERSION,
    release_tag=LEGACY_RELEASE_TAG,
    checkpoint_filename=LEGACY_CHECKPOINT_FILENAME,
    pca_filename=LEGACY_PCA_FILENAME,
    esm_filename=DEFAULT_ASSETS[0].filename,
    pca_dim=128,
    esm_model="esmc-600m-2024-12",
    training_description="Original paper reproduction bundle for PDBbind v2020 refined-set experiments.",
    assets=(
        DEFAULT_ASSETS[0],
        AssetSpec(
            "ProtCross checkpoint",
            LEGACY_CHECKPOINT_FILENAME,
            LEGACY_CHECKPOINT_URL,
            sha256="3eb6d8c9ef94541efc0444508e15d630c156a98e164a6caa08f2ae7a20371e45",
        ),
        AssetSpec(
            "ProtCross PCA reducer",
            LEGACY_PCA_FILENAME,
            LEGACY_PCA_URL,
            sha256="c4317684fb94c1337a44b844381d7e84472a6958b34a604b0f982984b629098b",
        ),
    ),
)


def get_asset_bundle(asset_version: str = "default") -> AssetBundle:
    normalized = asset_version.strip().lower()
    if normalized in {"default", "latest", DEFAULT_ASSET_VERSION, RELEASE_TAG.lower()}:
        return DEFAULT_ASSET_BUNDLE
    if normalized in {LEGACY_ASSET_VERSION, "0.1.1", LEGACY_RELEASE_TAG.lower()}:
        return LEGACY_ASSET_BUNDLE
    raise ValueError(
        "Unknown ProtCross asset version "
        f"{asset_version!r}; expected default, {DEFAULT_ASSET_VERSION}, {LEGACY_ASSET_VERSION}, or latest."
    )


def _infer_manifestless_bundle(assets_dir: Path, asset_version: str) -> AssetBundle:
    """Choose bundle filenames for a directory without a manifest.

    Old 0.1.1-era asset directories used the paper checkpoint/PCA filenames.
    Keeping this compatibility path makes upgraded source checkouts usable while
    still writing the concrete version into downstream summaries.
    """
    if asset_version.strip().lower() not in {"default", "latest", DEFAULT_ASSET_VERSION, RELEASE_TAG.lower()}:
        return get_asset_bundle(asset_version)
    legacy_files_present = (
        (assets_dir / LEGACY_CHECKPOINT_FILENAME).exists()
        and (assets_dir / LEGACY_PCA_FILENAME).exists()
        and not (assets_dir / DEFAULT_CHECKPOINT_FILENAME).exists()
        and not (assets_dir / DEFAULT_PCA_FILENAME).exists()
    )
    return LEGACY_ASSET_BUNDLE if legacy_files_present else DEFAULT_ASSET_BUNDLE


def get_default_assets_dir(asset_version: str = "default") -> Path:
    if os.environ.get("PROTCROSS_ASSETS_DIR"):
        return Path(os.environ["PROTCROSS_ASSETS_DIR"]).expanduser()
    bundle = get_asset_bundle(asset_version)
    return Path.home() / ".cache" / "protcross" / "assets" / bundle.release_tag


def setup_assets(
    output_dir: str | Path | None = None,
    *,
    asset_version: str = "default",
    esm_url: str | None = None,
    checkpoint_url: str | None = None,
    pca_url: str | None = None,
    force: bool = False,
    verify: bool = True,
    skip_esm: bool = False,
    accept_esm_license: bool = False,
) -> Path:
    """Download ProtCross assets and return the asset directory."""
    bundle = get_asset_bundle(asset_version)
    output_dir = Path(output_dir).expanduser() if output_dir else get_default_assets_dir(bundle.version)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        AssetSpec(
            bundle.assets[0].name,
            bundle.assets[0].filename,
            esm_url or bundle.assets[0].url,
            bundle.assets[0].sha256,
        ),
        AssetSpec(
            bundle.assets[1].name,
            bundle.assets[1].filename,
            checkpoint_url or bundle.assets[1].url,
            bundle.assets[1].sha256,
        ),
        AssetSpec(
            bundle.assets[2].name,
            bundle.assets[2].filename,
            pca_url or bundle.assets[2].url,
            bundle.assets[2].sha256,
        ),
    ]
    download_specs = specs[1:] if skip_esm else specs

    print(f"Installing ProtCross assets into {output_dir}")
    print("Note: ESM-C weights are distributed by EvolutionaryScale under their Hugging Face model terms.")
    if not skip_esm and _asset_needs_download(specs[0], output_dir / specs[0].filename, force=force, verify=verify):
        _require_esm_license_acceptance(accept_esm_license)
    for spec in download_specs:
        download_asset(spec, output_dir / spec.filename, force=force, verify=verify)

    write_env_file(output_dir, bundle=bundle, include_esm=(output_dir / bundle.esm_filename).exists())
    write_asset_manifest(output_dir, bundle, specs, include_esm=(output_dir / bundle.esm_filename).exists())
    print("\nAsset setup complete.")
    print("Use with: protcross predict input.pdb --out-dir protcross-results")
    print(f"Environment file written to: {output_dir / 'protcross.env'}")
    print(f"Asset manifest written to: {output_dir / ASSET_MANIFEST_FILENAME}")
    return output_dir


def download_asset(spec: AssetSpec, output_path: Path, *, force: bool = False, verify: bool = True) -> None:
    from tqdm import tqdm

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
    try:
        with _http_get(spec.url, stream=True, timeout=30) as response:
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
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty asset: {spec.filename}")

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


def _http_get(url: str, **kwargs):
    import requests

    return requests.get(url, **kwargs)


def write_env_file(output_dir: Path, *, bundle: AssetBundle | None = None, include_esm: bool = True) -> None:
    bundle = bundle or DEFAULT_ASSET_BUNDLE
    env_path = output_dir / "protcross.env"
    lines = [
        _export_line("PROTCROSS_ASSETS_DIR", output_dir),
        _export_line("PROTCROSS_ASSET_VERSION", bundle.version),
        _export_line("PROTCROSS_CHECKPOINT", output_dir / bundle.checkpoint_filename),
    ]
    if include_esm:
        lines.append(_export_line("PROTCROSS_ESM_WEIGHTS", output_dir / bundle.esm_filename))
    lines.extend(
        [
            _export_line("PROTCROSS_PCA", output_dir / bundle.pca_filename),
            "",
        ]
    )
    env_path.write_text("\n".join(lines), encoding="utf-8")


def write_asset_manifest(
    output_dir: Path,
    bundle: AssetBundle,
    specs: Iterable[AssetSpec],
    *,
    include_esm: bool = True,
) -> None:
    files = {}
    for spec in specs:
        if spec.filename == bundle.esm_filename and not include_esm:
            continue
        path = output_dir / spec.filename
        actual_sha256 = sha256_file(path) if path.exists() else None
        size_bytes = path.stat().st_size if path.exists() else None
        mtime_ns = path.stat().st_mtime_ns if path.exists() else None
        verified = None
        if spec.sha256 and actual_sha256:
            verified = actual_sha256 == spec.sha256
        files[spec.name] = {
            "filename": spec.filename,
            "source_url": spec.url,
            "url": spec.url,
            "release_tag": bundle.release_tag,
            "expected_sha256": spec.sha256,
            "actual_sha256": actual_sha256,
            "sha256": spec.sha256,
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
            "present": path.exists(),
            "verified": verified,
        }
    try:
        from protcross import __version__ as package_version
    except Exception:
        package_version = "unknown"

    payload = {
        "schema_version": "protcross-assets-v1",
        "package_version": package_version,
        "asset_version": bundle.version,
        "asset_bundle_version": bundle.version,
        "release_tag": bundle.release_tag,
        "checkpoint_filename": bundle.checkpoint_filename,
        "pca_filename": bundle.pca_filename,
        "esm_filename": bundle.esm_filename,
        "pca_dim": bundle.pca_dim,
        "esm_model": bundle.esm_model,
        "training_description": bundle.training_description,
        "files": files,
    }
    (output_dir / ASSET_MANIFEST_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_asset_manifest(assets_dir: str | Path) -> dict | None:
    manifest_path = Path(assets_dir) / ASSET_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_prediction_assets(
    ckpt_path: str | Path | None = None,
    esm_weights: str | Path | None = None,
    pca_path: str | Path | None = None,
    *,
    assets_dir: str | Path | None = None,
    auto_setup_assets: bool = True,
    asset_version: str = "default",
    refresh_assets: bool = False,
    offline: bool = False,
    accept_esm_license: bool = False,
    require_esm_license_for_use: bool = False,
    trust_unverified_assets: bool = False,
) -> ResolvedPredictionAssets:
    """Resolve prediction assets using the public precedence contract."""
    auto_setup_assets = auto_setup_assets and not offline
    ckpt, ckpt_source, ckpt_label = _path_or_env_with_source(ckpt_path, "PROTCROSS_CHECKPOINT", "--checkpoint")
    esm, esm_source, esm_label = _path_or_env_with_source(esm_weights, "PROTCROSS_ESM_WEIGHTS", "--esm-weights")
    pca, pca_source, pca_label = _path_or_env_with_source(pca_path, "PROTCROSS_PCA", "--pca")
    sources = {
        "checkpoint": ckpt_source,
        "esm_weights": esm_source,
        "pca": pca_source,
    }
    source_labels = {
        "checkpoint": ckpt_label,
        "esm_weights": esm_label,
        "pca": pca_label,
    }
    all_assets_explicit = all(source == "user" for source in sources.values())
    if all_assets_explicit:
        bundle = get_asset_bundle(asset_version)
        assert ckpt is not None and esm is not None and pca is not None
        assets = PredictorAssets(
            checkpoint=ckpt,
            esm_weights=esm,
            pca=pca,
            asset_version=bundle.version,
        )
    else:
        assets = (
            PredictorAssets.from_dir(assets_dir, asset_version=asset_version)
            if assets_dir
            else PredictorAssets.from_default_dir(asset_version=asset_version)
        )

    def fill_from_assets(*, require_exists: bool) -> None:
        nonlocal ckpt, esm, pca
        for name, candidate in (
            ("checkpoint", assets.checkpoint),
            ("esm_weights", assets.esm_weights),
            ("pca", assets.pca),
        ):
            if require_exists and not candidate.exists():
                continue
            if name == "checkpoint" and ckpt is None:
                ckpt = candidate
                sources[name] = "managed"
            elif name == "esm_weights" and esm is None:
                esm = candidate
                sources[name] = "managed"
            elif name == "pca" and pca is None:
                pca = candidate
                sources[name] = "managed"

    _raise_missing_user_assets(ckpt, esm, pca, sources, source_labels)
    fill_from_assets(require_exists=True)
    mismatches = _managed_asset_mismatches(assets, ckpt, esm, pca, sources)

    should_refresh_managed = refresh_assets and any(source != "user" for source in sources.values())
    if (should_refresh_managed or mismatches or _missing_asset_names(ckpt, esm, pca)) and auto_setup_assets:
        skip_esm = sources.get("esm_weights") == "user" and esm is not None and esm.exists()
        output_dir = setup_assets(
            assets_dir,
            asset_version=asset_version,
            force=should_refresh_managed,
            skip_esm=skip_esm,
            accept_esm_license=accept_esm_license,
        )
        assets = PredictorAssets.from_dir(output_dir, asset_version=asset_version)
        fill_from_assets(require_exists=True)
        mismatches = _managed_asset_mismatches(assets, ckpt, esm, pca, sources)

    missing = _missing_asset_names(ckpt, esm, pca)
    if missing:
        raise ValueError(
            "Missing prediction assets: "
            f"{', '.join(missing)}. Run `protcross setup-assets`, pass --assets-dir, "
            "or provide --checkpoint, --esm-weights, and --pca explicitly."
        )

    assert ckpt is not None and esm is not None and pca is not None
    missing_files = [path for path in (ckpt, esm, pca) if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Prediction asset file not found: "
            + ", ".join(str(path) for path in missing_files)
            + ". Use --refresh-assets to reinstall managed assets."
        )

    if not _trust_unverified_assets(trust_unverified_assets):
        untrusted = _unverified_user_assets(assets, ckpt, esm, pca, sources)
        if untrusted:
            raise RuntimeError(
                "Explicit prediction assets did not match the selected bundle SHA256: "
                + ", ".join(untrusted)
                + ". Use managed assets, pass the matching --asset-version, or set "
                "--trust-unverified-assets only for local files you trust."
            )

    if mismatches:
        raise RuntimeError(
            "Managed prediction assets failed SHA256 verification: "
            + ", ".join(mismatches)
            + ". Run `protcross setup-assets --force` or `protcross predict --refresh-assets`."
        )
    if require_esm_license_for_use:
        _require_esm_license_acceptance(accept_esm_license)

    return ResolvedPredictionAssets(
        checkpoint=ckpt,
        esm_weights=esm,
        pca=pca,
        assets=assets,
        asset_version=_resolved_asset_version(assets, ckpt, esm, pca, sources),
    )


def _path_or_env_with_source(
    value: str | Path | None,
    env_name: str,
    option_name: str,
) -> tuple[Path | None, str | None, str | None]:
    if value:
        return Path(value).expanduser(), "user", option_name
    env_value = os.environ.get(env_name)
    return (Path(env_value).expanduser(), "user", env_name) if env_value else (None, None, None)


def _raise_missing_user_assets(
    ckpt: Path | None,
    esm: Path | None,
    pca: Path | None,
    sources: dict[str, str | None],
    labels: dict[str, str | None],
) -> None:
    missing = [
        f"{name} from {labels[name]}: {path}"
        for name, path in (("checkpoint", ckpt), ("esm_weights", esm), ("pca", pca))
        if sources.get(name) == "user" and path is not None and not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Explicit prediction asset file not found: " + "; ".join(missing))


def _missing_asset_names(ckpt: Path | None, esm: Path | None, pca: Path | None) -> list[str]:
    return [
        name
        for name, value in (
            ("checkpoint", ckpt),
            ("esm_weights", esm),
            ("pca", pca),
        )
        if value is None
    ]


def _managed_asset_mismatches(
    assets: PredictorAssets,
    ckpt: Path | None,
    esm: Path | None,
    pca: Path | None,
    sources: dict[str, str | None],
) -> list[str]:
    bundle = get_asset_bundle(assets.asset_version)
    specs_by_filename = {spec.filename: spec for spec in bundle.assets if spec.sha256}
    paths = {
        "checkpoint": ckpt,
        "esm_weights": esm,
        "pca": pca,
    }
    mismatches = []
    for name, path in paths.items():
        if sources.get(name) != "managed" or path is None or not path.exists():
            continue
        spec = specs_by_filename.get(path.name)
        if spec is None or spec.sha256 is None:
            continue
        actual = sha256_file(path)
        if actual != spec.sha256:
            mismatches.append(f"{name} ({path.name})")
    return mismatches


def _unverified_user_assets(
    assets: PredictorAssets,
    ckpt: Path,
    esm: Path,
    pca: Path,
    sources: dict[str, str | None],
) -> list[str]:
    bundle = get_asset_bundle(assets.asset_version)
    expected = {
        "checkpoint": bundle.assets[1].sha256,
        "esm_weights": bundle.assets[0].sha256,
        "pca": bundle.assets[2].sha256,
    }
    paths = {"checkpoint": ckpt, "esm_weights": esm, "pca": pca}
    untrusted = []
    for name, path in paths.items():
        if sources.get(name) != "user":
            continue
        expected_sha = expected.get(name)
        if expected_sha and sha256_file(path) != expected_sha:
            untrusted.append(f"{name} ({path})")
    return untrusted


def _asset_needs_download(spec: AssetSpec, output_path: Path, *, force: bool, verify: bool) -> bool:
    if force or not output_path.exists():
        return True
    return bool(verify and spec.sha256 and sha256_file(output_path) != spec.sha256)


def _require_esm_license_acceptance(accept_esm_license: bool) -> None:
    if accept_esm_license or _env_truthy(ESM_LICENSE_ACCEPT_ENV):
        return
    raise RuntimeError(
        "ESM-C weights are distributed under EvolutionaryScale's model terms. "
        f"Review {ESM_LICENSE_URL} and rerun with --accept-esm-license "
        f"or {ESM_LICENSE_ACCEPT_ENV}=1 before downloading or using ESM-C."
    )


def require_esm_license_acceptance(accept_esm_license: bool) -> None:
    _require_esm_license_acceptance(accept_esm_license)


def _trust_unverified_assets(value: bool) -> bool:
    return value or _env_truthy(TRUST_UNVERIFIED_ASSETS_ENV)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolved_asset_version(
    assets: PredictorAssets,
    ckpt: Path,
    esm: Path,
    pca: Path,
    sources: dict[str, str | None],
) -> str:
    if any(source == "user" for source in sources.values()) and not _user_assets_match_selected_bundle(
        assets,
        ckpt,
        esm,
        pca,
        sources,
    ):
        return os.environ.get("PROTCROSS_ASSET_VERSION", "custom")
    return assets.asset_version


def _user_assets_match_selected_bundle(
    assets: PredictorAssets,
    ckpt: Path,
    esm: Path,
    pca: Path,
    sources: dict[str, str | None],
) -> bool:
    bundle = get_asset_bundle(assets.asset_version)
    expected = {
        "checkpoint": bundle.assets[1].sha256,
        "esm_weights": bundle.assets[0].sha256,
        "pca": bundle.assets[2].sha256,
    }
    paths = {"checkpoint": ckpt, "esm_weights": esm, "pca": pca}
    for name, path in paths.items():
        if sources.get(name) != "user":
            continue
        expected_sha = expected.get(name)
        if not expected_sha or not path.exists():
            return False
        try:
            actual_sha = sha256_file(path)
        except Exception:
            return False
        if actual_sha != expected_sha:
            return False
    return True


def _export_line(name: str, value: str | Path) -> str:
    return f"export {name}={shlex.quote(str(value))}"
