from __future__ import annotations

import hashlib
import json
import re
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
    tauri_config = json.loads(Path("desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    release_inputs = Path("desktop/installer/prepare_release_inputs.py").read_text(encoding="utf-8")
    mac_sign_script = Path("desktop/installer/sign-notarize-macos.sh").read_text(encoding="utf-8")
    mac_validate_script = Path("desktop/installer/validate-release-macos.sh").read_text(encoding="utf-8")

    assert "tauri:prepare-release" in package_json["scripts"]
    assert "prepare_release_inputs.py" in package_json["scripts"]["tauri:prepare-release"]
    assert "validate_runtime_bundle.py --runtime-dir ../runtime --backend cpu" in package_json["scripts"]["tauri:preflight"]
    assert package_json["scripts"]["tauri:release-build"].startswith("npm run tauri:prepare-release")
    assert 'default="cpu"' in release_inputs
    assert "Public installers bundle the CPU backend only" in release_inputs
    assert "cd frontend" in tauri_config["build"]["beforeBuildCommand"]
    assert "cd desktop/frontend" in tauri_config["build"]["beforeBuildCommand"]
    assert "cd ../frontend" in tauri_config["build"]["beforeBuildCommand"]
    assert tauri_config["bundle"]["category"] == "Education"
    assert "--backend cpu" in mac_sign_script
    assert "--backend cpu" in mac_validate_script
    assert "--backend all" not in mac_sign_script
    assert "--backend all" not in mac_validate_script


def test_readme_frontloads_desktop_download_links():
    readme = Path("README.md").read_text(encoding="utf-8")
    quick_start = readme.split("## Table of Contents", 1)[0]

    assert "### Recommended for new users: Desktop app" in quick_start
    assert "https://github.com/GeraltZeroZhong/ProtCross/releases/download/v0.2.0/ProtCross%20Desktop_0.2.0_x64-setup.exe" in quick_start
    assert "https://github.com/GeraltZeroZhong/ProtCross/releases/download/v0.2.0/ProtCross_Desktop_0.2.0_macos.dmg" in quick_start
    assert quick_start.index("### Recommended for new users: Desktop app") < quick_start.index("### Command line")
    assert "official ProtCross release assets" in quick_start
    assert "unsigned testing builds" in quick_start


def test_packaged_sidecar_smoke_uses_env_token_only():
    validator = Path("desktop/installer/validate_packaged_sidecar.py").read_text(encoding="utf-8")

    assert "PROTCROSS_DESKTOP_TOKEN" in validator
    assert '"--token"' not in validator


def test_windows_release_scripts_validate_signed_runtime_and_installed_package():
    sign_script = Path("desktop/installer/sign-installer.ps1").read_text(encoding="utf-8")
    validate_script = Path("desktop/installer/validate-release.ps1").read_text(encoding="utf-8")
    nsis_hook = Path("desktop/installer/esm-license-confirmation.nsh").read_text(encoding="utf-8")
    cpu_installer = Path("desktop/runtime/install_cpu_backend.ps1").read_text(encoding="utf-8")
    gpu_installer = Path("desktop/runtime/install_gpu_backend.ps1").read_text(encoding="utf-8")
    tauri_main = Path("desktop/src-tauri/src/main.rs").read_text(encoding="utf-8")

    assert "uv\\uv.exe" in sign_script
    assert "uv.sha256" in sign_script
    assert "Set-AuthenticodeSignature" in sign_script
    assert "ValidateInstalledPackage" in validate_script
    assert "RequireInstallerSignature" in validate_script
    assert "AcceptEsmLicenseForCi" in validate_script
    assert "PROTCROSS_DESKTOP_CI_ACCEPT_ESMC_LICENSE" in validate_script
    assert 'validate_runtime_bundle.py"), "--runtime-dir", $RuntimeDir, "--backend", "cpu"' in validate_script
    assert "validate_packaged_sidecar.py" in validate_script
    assert "install_cpu_backend.ps1" in validate_script
    assert "IfSilent" in nsis_hook
    assert "PROTCROSS_DESKTOP_CI_ACCEPT_ESMC_LICENSE" in nsis_hook
    for script in (validate_script, cpu_installer, gpu_installer):
        assert "${LASTEXITCODE}:" in script
        assert "$LASTEXITCODE:" not in script
    assert '"-AllowOnlinePackageIndex"' in tauri_main
    assert '"--allow-online-package-index"' in tauri_main


def test_release_scripts_use_cross_platform_hashing_and_wheel_builds():
    bundled_assets_script = Path("desktop/installer/prepare_bundled_assets.sh").read_text(encoding="utf-8")
    wheelhouse_script = Path("desktop/installer/prepare_runtime_wheelhouse.py").read_text(encoding="utf-8")
    ci_workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    cargo_toml = Path("desktop/src-tauri/Cargo.toml").read_text(encoding="utf-8")
    rust_toolchain = Path("rust-toolchain.toml").read_text(encoding="utf-8")

    assert "hashlib.sha256" in bundled_assets_script
    assert "sha256sum" not in bundled_assets_script
    assert '"wheel"' in wheelhouse_script
    assert '"download"' not in wheelhouse_script
    assert "--only-binary=:all:" not in wheelhouse_script
    assert "local_desktop_backend_wheel" in wheelhouse_script
    assert 'shutil.rmtree(package_dir / "build"' in wheelhouse_script
    assert 'features = ["protocol-asset"]' in cargo_toml
    assert 'channel = "1.88.0"' in rust_toolchain
    assert "cargo +1.88.0 check --locked" in ci_workflow
    assert "dtolnay/rust-toolchain@1.88.0" in ci_workflow
    assert "--backend cpu" in ci_workflow
    assert "-AcceptEsmLicenseForCi" in ci_workflow
    assert "tauri:release-build -- --bundles app -- --locked" in ci_workflow
    assert "tauri:release-build -- --bundles nsis -- --locked" in ci_workflow


def test_tauri_lockfile_pins_release_compatible_family():
    cargo_lock = Path("desktop/src-tauri/Cargo.lock").read_text(encoding="utf-8")
    package_json = json.loads(Path("desktop/frontend/package.json").read_text(encoding="utf-8"))

    package_versions = {
        name: version
        for name, version in re.findall(
            r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"',
            cargo_lock,
        )
    }

    assert package_versions["tauri"] == "2.11.4"
    assert package_versions["tauri-build"] == "2.6.3"
    assert package_versions["tauri-codegen"] == "2.6.3"
    assert package_versions["tauri-macros"] == "2.6.3"
    assert package_versions["tauri-plugin"] == "2.6.3"
    assert package_versions["tauri-plugin-dialog"] == "2.7.1"
    assert package_versions["tauri-plugin-fs"] == "2.5.1"
    assert package_versions["tauri-runtime"] == "2.11.3"
    assert package_versions["tauri-runtime-wry"] == "2.11.4"
    assert package_versions["tauri-utils"] == "2.9.3"
    assert package_versions["time"] == "0.3.51"
    assert package_versions["time-macros"] == "0.2.30"
    assert package_json["dependencies"]["@tauri-apps/api"] == "2.11.1"
    assert package_json["dependencies"]["@tauri-apps/plugin-dialog"] == "2.7.1"
    assert package_json["devDependencies"]["@tauri-apps/cli"] == "2.11.4"


def test_tauri_default_icon_exists_for_generate_context():
    png_icon = Path("desktop/src-tauri/icons/icon.png")
    windows_icon = Path("desktop/src-tauri/icons/icon.ico")

    assert png_icon.exists()
    png_data = png_icon.read_bytes()
    assert png_data.startswith(b"\x89PNG\r\n\x1a\n")
    assert png_data[12:16] == b"IHDR"

    assert windows_icon.exists()
    ico_data = windows_icon.read_bytes()
    assert ico_data[:6] == b"\x00\x00\x01\x00\x01\x00"
    assert ico_data[22:].startswith(b"\x89PNG\r\n\x1a\n")


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
