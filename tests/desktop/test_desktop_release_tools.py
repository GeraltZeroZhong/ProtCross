from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_validate_runtime_bundle_accepts_hash_locked_fixture(tmp_path: Path):
    runtime_dir = _make_runtime_bundle(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "desktop/installer/validate_runtime_bundle.py",
            "--runtime-dir",
            str(runtime_dir),
            "--backend",
            "all",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_validate_runtime_bundle_rejects_missing_backend_hash(tmp_path: Path):
    runtime_dir = _make_runtime_bundle(tmp_path)
    (runtime_dir / "requirements-gpu.hashes").unlink()

    result = subprocess.run(
        [
            sys.executable,
            "desktop/installer/validate_runtime_bundle.py",
            "--runtime-dir",
            str(runtime_dir),
            "--backend",
            "all",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "requirements-gpu.hashes" in result.stderr


def test_validate_packaged_sidecar_accepts_resource_layout(tmp_path: Path):
    resource_dir = tmp_path / "resources"
    (resource_dir / "backend" / "protcross_desktop").mkdir(parents=True)
    (resource_dir / "backend" / "protcross_desktop" / "server.py").write_text("", encoding="utf-8")
    (resource_dir / "python-src" / "protcross").mkdir(parents=True)
    (resource_dir / "python-src" / "protcross" / "__init__.py").write_text("", encoding="utf-8")
    (resource_dir / "runtime").mkdir()
    (resource_dir / "bundled-assets").mkdir()
    (resource_dir / "bundled-assets" / "protcross-desktop-bundled-assets.json").write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "desktop/installer/validate_packaged_sidecar.py",
            "--resource-dir",
            str(resource_dir),
            "--no-start",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_frontend_release_build_prepares_generated_inputs():
    package_json = json.loads(Path("desktop/frontend/package.json").read_text(encoding="utf-8"))

    assert "tauri:prepare-release" in package_json["scripts"]
    assert "prepare_release_inputs.py" in package_json["scripts"]["tauri:prepare-release"]
    assert package_json["scripts"]["tauri:release-build"].startswith("npm run tauri:prepare-release")


def test_windows_release_scripts_validate_signed_runtime_and_installed_package():
    sign_script = Path("desktop/installer/sign-installer.ps1").read_text(encoding="utf-8")
    validate_script = Path("desktop/installer/validate-release.ps1").read_text(encoding="utf-8")

    assert "uv\\uv.exe" in sign_script
    assert "uv.sha256" in sign_script
    assert "Set-AuthenticodeSignature" in sign_script
    assert "ValidateInstalledPackage" in validate_script
    assert "RequireInstallerSignature" in validate_script
    assert "validate_packaged_sidecar.py" in validate_script
    assert "install_cpu_backend.ps1" in validate_script


def _make_runtime_bundle(root: Path) -> Path:
    runtime_dir = root / "runtime"
    wheelhouse = runtime_dir / "wheelhouse"
    uv_dir = runtime_dir / "uv"
    wheelhouse.mkdir(parents=True)
    uv_dir.mkdir(parents=True)
    (runtime_dir / "requirements-common.lock").write_text("protcross==0.1.3\n", encoding="utf-8")
    wheels = {
        "protcross": wheelhouse / "protcross-0.1.3-py3-none-any.whl",
        "protcross-desktop-backend": wheelhouse / "protcross_desktop_backend-0.2.0-py3-none-any.whl",
        "torch": wheelhouse / "torch-2.3.1-cp310-cp310-win_amd64.whl",
    }
    for name, wheel in wheels.items():
        wheel.write_bytes(f"fake {name} wheel\n".encode("utf-8"))
    for backend in ("cpu", "gpu"):
        (runtime_dir / f"requirements-{backend}.hashes").write_text(
            "\n".join(
                [
                    f"protcross==0.1.3 --hash=sha256:{_sha256(wheels['protcross'])}",
                    (
                        "protcross-desktop-backend==0.2.0 "
                        f"--hash=sha256:{_sha256(wheels['protcross-desktop-backend'])}"
                    ),
                    f"torch==2.3.1 --hash=sha256:{_sha256(wheels['torch'])}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    uv = uv_dir / ("uv.exe" if sys.platform == "win32" else "uv")
    uv.write_bytes(b"fake uv\n")
    uv.chmod(0o755)
    (uv_dir / "uv.sha256").write_text(f"{_sha256(uv)}  {uv.name}\n", encoding="utf-8")
    return runtime_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
