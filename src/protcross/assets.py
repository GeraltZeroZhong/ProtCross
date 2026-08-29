"""Download and validate ProtCross runtime assets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Callable
import uuid


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
    size_bytes: int | None = None


class DownloadCancelled(RuntimeError):
    """Raised when an asset download is cancelled without deleting partial data."""


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
            manifest_version = str(manifest.get("asset_version", asset_version))
            bundle = get_asset_bundle(manifest_version)
            if asset_version.strip().lower() not in {"default", "latest"}:
                requested = get_asset_bundle(asset_version)
                if requested.version != bundle.version:
                    raise ValueError(
                        "Asset directory version conflict: "
                        f"requested {requested.version}, but {assets_dir / ASSET_MANIFEST_FILENAME} "
                        f"records {bundle.version}."
                    )
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
    asset_metadata: dict[str, object]


DEFAULT_ASSETS = (
    AssetSpec(
        name="ESM-C 600M weights",
        filename="esmc_600m_2024_12_v0.pth",
        url=DEFAULT_ESM_URL,
        sha256="8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324",
        size_bytes=2_300_275_866,
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
    training_description=(
        "Binding MOAD-selected structures with heuristic hetero-residue-proximity labels; "
        "release training used no held-out validation or test split."
    ),
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
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else get_default_assets_dir(bundle.version).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        existing_manifest = read_asset_manifest(output_dir) or {}
    except RuntimeError:
        if not force:
            raise
        manifest_path = output_dir / ASSET_MANIFEST_FILENAME
        quarantine_path = output_dir / f"{ASSET_MANIFEST_FILENAME}.corrupt-{uuid.uuid4().hex}"
        manifest_path.replace(quarantine_path)
        print(f"[warn] Quarantined unreadable asset manifest: {quarantine_path}")
        existing_manifest = {}
    license_accepted = bool(
        accept_esm_license
        or _env_truthy(ESM_LICENSE_ACCEPT_ENV)
        or existing_manifest.get("esm_license", {}).get("accepted") is True
    )
    specs = [
        AssetSpec(
            bundle.assets[0].name,
            bundle.assets[0].filename,
            esm_url or bundle.assets[0].url,
            bundle.assets[0].sha256,
            bundle.assets[0].size_bytes,
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
    known_sha256: dict[str, str] = {}
    esm_path = output_dir / specs[0].filename
    esm_needs_download = not skip_esm and (force or not esm_path.exists())
    if not skip_esm and not esm_needs_download and verify and specs[0].sha256:
        known_sha256[specs[0].filename] = sha256_file(esm_path)
        esm_needs_download = known_sha256[specs[0].filename] != specs[0].sha256
    if esm_needs_download:
        _require_esm_license_acceptance(license_accepted)
    resolved_sha256: dict[str, str] = {}
    for spec in download_specs:
        actual_sha256 = download_asset(
            spec,
            output_dir / spec.filename,
            force=force,
            verify=verify,
            known_sha256=known_sha256.get(spec.filename),
        )
        if actual_sha256:
            resolved_sha256[spec.filename] = actual_sha256

    write_env_file(output_dir, bundle=bundle, include_esm=(output_dir / bundle.esm_filename).exists())
    write_asset_manifest(
        output_dir,
        bundle,
        specs,
        include_esm=(output_dir / bundle.esm_filename).exists(),
        esm_license_accepted=license_accepted,
        previous_manifest=existing_manifest,
        actual_sha256_by_filename=resolved_sha256,
    )
    print("\nAsset setup complete.")
    print("Use with: protcross predict input.pdb --out-dir protcross-results")
    print(f"Environment file written to: {output_dir / 'protcross.env'}")
    print(f"Asset manifest written to: {output_dir / ASSET_MANIFEST_FILENAME}")
    return output_dir


def download_asset(
    spec: AssetSpec,
    output_path: Path,
    *,
    force: bool = False,
    verify: bool = True,
    progress_callback: Callable[[int, int | None], None] | None = None,
    cancel_event: Event | None = None,
    known_sha256: str | None = None,
) -> str | None:
    """Download and verify an asset, resuming a retained ``.part`` file when possible."""
    output_path = Path(output_path).expanduser()
    with _asset_download_lock(output_path):
        return _download_asset_locked(
            spec,
            output_path,
            force=force,
            verify=verify,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            known_sha256=known_sha256,
        )


def _download_asset_locked(
    spec: AssetSpec,
    output_path: Path,
    *,
    force: bool,
    verify: bool,
    progress_callback: Callable[[int, int | None], None] | None,
    cancel_event: Event | None,
    known_sha256: str | None,
) -> str | None:
    from tqdm import tqdm

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        actual_sha256 = known_sha256
        if verify and spec.sha256 and actual_sha256 is None:
            actual_sha256 = sha256_file(output_path)
        if not verify or not spec.sha256 or actual_sha256 == spec.sha256:
            if progress_callback:
                size = output_path.stat().st_size
                progress_callback(size, spec.size_bytes or size)
            print(f"[skip] {spec.name}: {output_path}")
            return actual_sha256
        print(f"[warn] Existing file failed SHA256 verification and will be replaced: {output_path}")

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    if force and tmp_path.exists():
        tmp_path.unlink()

    print(f"[download] {spec.name}")
    print(f"           {spec.url}")
    resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
    if spec.size_bytes and resume_from > spec.size_bytes:
        tmp_path.unlink()
        resume_from = 0
    # A process may have completed the file just before interruption.  A hash
    # proves completeness even when the remote server did not publish a size.
    complete_by_hash = bool(
        resume_from
        and verify
        and spec.sha256
        and sha256_file(tmp_path) == spec.sha256
    )
    if complete_by_hash or (spec.size_bytes and resume_from == spec.size_bytes):
        complete = complete_by_hash or not verify or not spec.sha256
        if complete:
            tmp_path.replace(output_path)
            if progress_callback:
                progress_callback(resume_from, spec.size_bytes or resume_from)
            print(f"[ok] {output_path}")
            return spec.sha256 if complete_by_hash else None
        tmp_path.unlink()
        resume_from = 0
    _require_download_space(output_path.parent, spec.size_bytes, already_downloaded=resume_from)
    request_headers = {"Range": f"bytes={resume_from}-"} if resume_from else None
    total: int | None = spec.size_bytes
    try:
        for attempt in range(2):
            with _http_get(spec.url, stream=True, timeout=30, headers=request_headers) as response:
                status_code = int(getattr(response, "status_code", 200))
                if resume_from and status_code == 416 and attempt == 0:
                    # The server considers the retained range complete or
                    # invalid. A valid complete file was already promoted
                    # above; restart a corrupt/unknown partial once from zero.
                    tmp_path.unlink(missing_ok=True)
                    resume_from = 0
                    request_headers = None
                    continue
                response.raise_for_status()
                resumed = bool(resume_from and status_code == 206)
                if resumed and _content_range_start(response.headers.get("content-range")) != resume_from:
                    # Never append an unverified range. Retry once without a
                    # Range header; some proxies strip or rewrite this header.
                    if attempt == 0:
                        resume_from = 0
                        request_headers = None
                        continue
                    raise RuntimeError("Server returned an invalid Content-Range for the requested asset.")
                if resume_from and not resumed:
                    # The server ignored Range. Restart safely instead of
                    # appending duplicate bytes to the retained partial.
                    resume_from = 0
                response_bytes = int(response.headers.get("content-length", 0) or 0)
                if spec.size_bytes and response_bytes > spec.size_bytes - resume_from:
                    raise RuntimeError(
                        f"Server response exceeds the expected size for {spec.filename}."
                    )
                total = spec.size_bytes or ((resume_from + response_bytes) if response_bytes > 0 else None)
                mode = "ab" if resumed else "wb"
                downloaded = resume_from
                if progress_callback:
                    progress_callback(resume_from, total)
                with tmp_path.open(mode) as file, tqdm(
                    total=total,
                    initial=resume_from,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=spec.filename,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if cancel_event is not None and cancel_event.is_set():
                            raise DownloadCancelled(
                                f"Download cancelled; partial data retained at {tmp_path} for resume."
                            )
                        if not chunk:
                            continue
                        if spec.size_bytes and downloaded + len(chunk) > spec.size_bytes:
                            file.truncate(resume_from)
                            raise RuntimeError(
                                f"Server response exceeds the expected size for {spec.filename}."
                            )
                        file.write(chunk)
                        downloaded += len(chunk)
                        progress.update(len(chunk))
                        if progress_callback:
                            progress_callback(downloaded, total)
                break
    except DownloadCancelled:
        raise
    except Exception as exc:
        # Keep a non-empty partial file so transient network failures do not waste gigabytes.
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            tmp_path.unlink(missing_ok=True)
        if tmp_path.exists():
            raise RuntimeError(
                f"Download interrupted; partial data retained at {tmp_path}. Rerun the same command to resume."
            ) from exc
        raise

    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty asset: {spec.filename}")

    if spec.size_bytes and tmp_path.stat().st_size != spec.size_bytes:
        raise RuntimeError(
            f"Download interrupted at {tmp_path.stat().st_size} of {spec.size_bytes} bytes; "
            f"partial data retained at {tmp_path}. Rerun the same command to resume."
        )

    actual_sha256 = None
    if verify and spec.sha256:
        actual_sha256 = sha256_file(tmp_path)
        if actual_sha256 != spec.sha256:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch for {spec.filename}: expected {spec.sha256}, got {actual_sha256}"
            )

    tmp_path.replace(output_path)
    if progress_callback:
        progress_callback(output_path.stat().st_size, total)
    print(f"[ok] {output_path}")
    return actual_sha256


def _content_range_start(value: str | None) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes\s+(\d+)-\d+/(?:\d+|\*)", value.strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


@contextmanager
def _asset_download_lock(output_path: Path):
    """Serialize writers to one asset across processes without extra dependencies."""
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _require_download_space(directory: Path, expected_size: int | None, *, already_downloaded: int) -> None:
    if not expected_size:
        return
    remaining = max(0, expected_size - already_downloaded)
    reserve = 256 * 1024 * 1024
    free = shutil.disk_usage(directory).free
    if free < remaining + reserve:
        raise OSError(
            "Not enough free disk space for this asset: "
            f"need about {(remaining + reserve) / (1024 ** 3):.2f} GiB including safety margin, "
            f"but only {free / (1024 ** 3):.2f} GiB is available in {directory}."
        )


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
    _write_text_atomic(env_path, "\n".join(lines), encoding="utf-8")
    values = {
        "PROTCROSS_ASSETS_DIR": output_dir,
        "PROTCROSS_ASSET_VERSION": bundle.version,
        "PROTCROSS_CHECKPOINT": output_dir / bundle.checkpoint_filename,
        "PROTCROSS_PCA": output_dir / bundle.pca_filename,
    }
    if include_esm:
        values["PROTCROSS_ESM_WEIGHTS"] = output_dir / bundle.esm_filename
    ps_lines = [f"$env:{name} = {_powershell_quote(str(value))}" for name, value in values.items()]
    _write_text_atomic(output_dir / "protcross.env.ps1", "\n".join(ps_lines) + "\n", encoding="utf-8")
    cmd_lines = [f'set "{name}={value}"' for name, value in values.items()]
    _write_text_atomic(output_dir / "protcross.env.cmd", "\r\n".join(cmd_lines) + "\r\n", encoding="utf-8")


def write_asset_manifest(
    output_dir: Path,
    bundle: AssetBundle,
    specs: Iterable[AssetSpec],
    *,
    include_esm: bool = True,
    esm_license_accepted: bool = False,
    previous_manifest: dict | None = None,
    actual_sha256_by_filename: dict[str, str] | None = None,
) -> None:
    actual_sha256_by_filename = actual_sha256_by_filename or {}
    files = {}
    for spec in specs:
        if spec.filename == bundle.esm_filename and not include_esm:
            continue
        path = output_dir / spec.filename
        actual_sha256 = (
            actual_sha256_by_filename.get(spec.filename)
            or (sha256_file(path) if path.exists() else None)
        )
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
            "expected_size_bytes": spec.size_bytes,
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
        "esm_license": {
            "accepted": bool(esm_license_accepted),
            "url": ESM_LICENSE_URL,
            "accepted_at": (
                (previous_manifest or {}).get("esm_license", {}).get("accepted_at")
                if (previous_manifest or {}).get("esm_license", {}).get("accepted") is True
                else datetime.now(timezone.utc).isoformat(timespec="seconds") if esm_license_accepted else None
            ),
        },
        "files": files,
    }
    _write_text_atomic(
        output_dir / ASSET_MANIFEST_FILENAME,
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _write_text_atomic(path: Path, text: str, *, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding=encoding, newline="") as handle:
            handle.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_asset_manifest(assets_dir: str | Path) -> dict | None:
    manifest_path = Path(assets_dir) / ASSET_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Asset manifest is unreadable: {manifest_path}. "
            "Run `protcross setup-assets --refresh-assets` to quarantine and rebuild it."
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Asset manifest must contain a JSON object: {manifest_path}. "
            "Run `protcross setup-assets --refresh-assets` to quarantine and rebuild it."
        )
    return payload


def build_prediction_asset_metadata(
    checkpoint: str | Path,
    esm_weights: str | Path,
    pca: str | Path,
    *,
    asset_version: str | None,
    selected_bundle_version: str | None = None,
    sources: dict[str, str | None] | None = None,
    source_labels: dict[str, str | None] | None = None,
    actual_sha256: dict[str, str] | None = None,
    trust_unverified_assets: bool = False,
) -> dict[str, object]:
    """Describe the exact runtime assets recorded in prediction outputs.

    ``verified`` means that the actual file hash matches the selected release
    bundle. A user trust override is recorded separately and never converted
    into a successful verification result.
    """
    paths = {
        "checkpoint": Path(checkpoint).expanduser(),
        "esm_weights": Path(esm_weights).expanduser(),
        "pca": Path(pca).expanduser(),
    }
    bundle = None
    bundle_name = selected_bundle_version or asset_version
    if bundle_name:
        try:
            bundle = get_asset_bundle(bundle_name)
        except ValueError:
            bundle = None
    expected_sha256 = {name: None for name in paths}
    if bundle is not None:
        specs_by_filename = {spec.filename: spec for spec in bundle.assets}
        expected_sha256 = {
            "checkpoint": specs_by_filename[bundle.checkpoint_filename].sha256,
            "esm_weights": specs_by_filename[bundle.esm_filename].sha256,
            "pca": specs_by_filename[bundle.pca_filename].sha256,
        }

    sources = sources or {}
    source_labels = source_labels or {}
    actual_sha256 = actual_sha256 or {}
    entries: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        actual = actual_sha256.get(name) or sha256_file(path)
        expected = expected_sha256[name]
        verified = actual == expected if expected else None
        source = sources.get(name) or "direct"
        trust_override = bool(
            verified is False and source == "user" and trust_unverified_assets
        )
        if verified is True:
            verification_status = "verified_release_sha256"
        elif trust_override:
            verification_status = "sha256_mismatch_trusted_by_user"
        elif verified is False:
            verification_status = "sha256_mismatch"
        else:
            verification_status = "no_release_sha256_available"
        entries[name] = {
            "path": str(path.resolve()),
            "filename": path.name,
            "source": source,
            "source_label": source_labels.get(name) or _default_asset_source_label(source),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "sha256": actual,
            "verified": verified,
            "verification_status": verification_status,
            "trust_unverified_override": trust_override,
            "size_bytes": path.stat().st_size,
        }

    return {
        "asset_version": asset_version,
        "selected_bundle_version": bundle.version if bundle is not None else bundle_name,
        "all_assets_verified": all(entry["verified"] is True for entry in entries.values()),
        "contains_unverified_assets": any(entry["verified"] is not True for entry in entries.values()),
        **entries,
    }


def _default_asset_source_label(source: str) -> str:
    return {
        "managed": "managed asset directory",
        "user": "explicit path or environment variable",
        "predictor_assets": "PredictorAssets",
        "direct": "direct from_files path",
    }.get(source, source)


def _asset_sha256(path: Path, cache: dict[Path, str] | None = None) -> str:
    cache_key = path.expanduser().resolve()
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    actual = sha256_file(path)
    if cache is not None:
        cache[cache_key] = actual
    return actual


def resolve_prediction_assets(
    ckpt_path: str | Path | None = None,
    esm_weights: str | Path | None = None,
    pca_path: str | Path | None = None,
    *,
    assets_dir: str | Path | None = None,
    auto_setup_assets: bool = True,
    asset_version: str | None = None,
    refresh_assets: bool = False,
    offline: bool = False,
    accept_esm_license: bool = False,
    require_esm_license_for_use: bool = False,
    trust_unverified_assets: bool = False,
) -> ResolvedPredictionAssets:
    """Resolve prediction assets using the public precedence contract."""
    if refresh_assets and (offline or not auto_setup_assets):
        raise ValueError(
            "refresh_assets requires automatic asset setup and cannot be combined with offline mode."
        )
    auto_setup_assets = auto_setup_assets and not offline
    hash_cache: dict[Path, str] = {}
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
    # An explicit API/CLI value wins. When omitted, generated environment files
    # can select their matching bundle before falling back to the current one.
    # Thus an explicit ``"default"`` is never replaced by stale environment state.
    asset_version = asset_version or os.environ.get("PROTCROSS_ASSET_VERSION") or "default"
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
    _seed_hash_cache_from_manifest(assets, sources, hash_cache)
    mismatches = _managed_asset_mismatches(
        assets,
        ckpt,
        esm,
        pca,
        sources,
        hash_cache=hash_cache,
    )

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
        hash_cache.clear()
        _seed_hash_cache_from_manifest(assets, sources, hash_cache)
        mismatches = _managed_asset_mismatches(
            assets,
            ckpt,
            esm,
            pca,
            sources,
            hash_cache=hash_cache,
        )

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

    trust_unverified = _trust_unverified_assets(trust_unverified_assets)
    if not trust_unverified:
        untrusted = _unverified_user_assets(
            assets,
            ckpt,
            esm,
            pca,
            sources,
            hash_cache=hash_cache,
        )
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
        persisted_acceptance = _asset_manifest_accepts_esm_license(assets)
        current_acceptance = accept_esm_license or _env_truthy(ESM_LICENSE_ACCEPT_ENV)
        _require_esm_license_acceptance(current_acceptance or persisted_acceptance)
        if current_acceptance and not persisted_acceptance:
            _persist_esm_license_acceptance(assets)

    resolved_asset_version = _resolved_asset_version(
        assets,
        ckpt,
        esm,
        pca,
        sources,
        hash_cache=hash_cache,
    )
    paths = {"checkpoint": ckpt, "esm_weights": esm, "pca": pca}
    actual_sha256 = {
        name: _asset_sha256(path, hash_cache)
        for name, path in paths.items()
    }
    asset_metadata = build_prediction_asset_metadata(
        ckpt,
        esm,
        pca,
        asset_version=resolved_asset_version,
        selected_bundle_version=assets.asset_version,
        sources=sources,
        source_labels=source_labels,
        actual_sha256=actual_sha256,
        trust_unverified_assets=trust_unverified,
    )
    return ResolvedPredictionAssets(
        checkpoint=ckpt,
        esm_weights=esm,
        pca=pca,
        assets=assets,
        asset_version=resolved_asset_version,
        asset_metadata=asset_metadata,
    )


def _seed_hash_cache_from_manifest(
    assets: PredictorAssets,
    sources: dict[str, str | None],
    hash_cache: dict[Path, str],
) -> None:
    """Reuse verified managed hashes while the recorded files are unchanged."""
    if not any(source == "managed" for source in sources.values()):
        return
    manifest = read_asset_manifest(assets.checkpoint.parent)
    if not manifest or manifest.get("asset_version") != assets.asset_version:
        return

    bundle = get_asset_bundle(assets.asset_version)
    specs = {
        "checkpoint": next(spec for spec in bundle.assets if spec.filename == bundle.checkpoint_filename),
        "esm_weights": next(spec for spec in bundle.assets if spec.filename == bundle.esm_filename),
        "pca": next(spec for spec in bundle.assets if spec.filename == bundle.pca_filename),
    }
    paths = {
        "checkpoint": assets.checkpoint,
        "esm_weights": assets.esm_weights,
        "pca": assets.pca,
    }
    entries = {
        str(entry.get("filename")): entry
        for entry in (manifest.get("files") or {}).values()
        if isinstance(entry, dict) and entry.get("filename")
    }
    for name, path in paths.items():
        if sources.get(name) != "managed" or not path.exists():
            continue
        spec = specs[name]
        entry = entries.get(path.name)
        if not entry or not spec.sha256:
            continue
        stat = path.stat()
        if (
            entry.get("expected_sha256") == spec.sha256
            and entry.get("actual_sha256") == spec.sha256
            and entry.get("verified") is True
            and entry.get("size_bytes") == stat.st_size
            and entry.get("mtime_ns") == stat.st_mtime_ns
        ):
            hash_cache[path.expanduser().resolve()] = spec.sha256


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
    *,
    hash_cache: dict[Path, str] | None = None,
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
        actual = _asset_sha256(path, hash_cache)
        if actual != spec.sha256:
            mismatches.append(f"{name} ({path.name})")
    return mismatches


def _unverified_user_assets(
    assets: PredictorAssets,
    ckpt: Path,
    esm: Path,
    pca: Path,
    sources: dict[str, str | None],
    *,
    hash_cache: dict[Path, str] | None = None,
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
        if expected_sha and _asset_sha256(path, hash_cache) != expected_sha:
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
    *,
    hash_cache: dict[Path, str] | None = None,
) -> str:
    if any(source == "user" for source in sources.values()) and not _user_assets_match_selected_bundle(
        assets,
        ckpt,
        esm,
        pca,
        sources,
        hash_cache=hash_cache,
    ):
        return os.environ.get("PROTCROSS_ASSET_VERSION", "custom")
    return assets.asset_version


def _user_assets_match_selected_bundle(
    assets: PredictorAssets,
    ckpt: Path,
    esm: Path,
    pca: Path,
    sources: dict[str, str | None],
    *,
    hash_cache: dict[Path, str] | None = None,
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
            actual_sha = _asset_sha256(path, hash_cache)
        except Exception:
            return False
        if actual_sha != expected_sha:
            return False
    return True


def _export_line(name: str, value: str | Path) -> str:
    return f"export {name}={shlex.quote(str(value))}"


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _asset_manifest_accepts_esm_license(assets: PredictorAssets) -> bool:
    manifest = read_asset_manifest(assets.esm_weights.parent)
    return bool(manifest and manifest.get("esm_license", {}).get("accepted") is True)


def _persist_esm_license_acceptance(assets: PredictorAssets) -> bool:
    """Persist explicit CLI acceptance when an existing managed manifest is reused."""
    manifest_path = assets.esm_weights.parent / ASSET_MANIFEST_FILENAME
    manifest = read_asset_manifest(assets.esm_weights.parent)
    if not manifest:
        return False
    manifest["esm_license"] = {
        "accepted": True,
        "url": ESM_LICENSE_URL,
        "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        from protcross import __version__ as package_version
    except Exception:
        package_version = manifest.get("package_version", "unknown")
    manifest["package_version"] = package_version
    try:
        _write_text_atomic(manifest_path, json.dumps(manifest, indent=2), encoding="utf-8")
    except OSError:
        return False
    return True
