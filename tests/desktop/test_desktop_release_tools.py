from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_validate_runtime_bundle_rejects_stale_protcross_wheels(tmp_path: Path):
    expected = _core_version()
    stale = "0.2.0" if expected != "0.2.0" else "0.1.9"
    runtime_dir = _make_runtime_bundle(tmp_path, version=stale)

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
    assert f"expected protcross=={expected}" in result.stderr
    assert f"expected protcross-desktop-backend=={expected}" in result.stderr


def test_validate_runtime_bundle_rejects_unreferenced_stale_release_wheel(tmp_path: Path):
    expected = _core_version()
    stale = "0.2.0" if expected != "0.2.0" else "0.1.9"
    runtime_dir = _make_runtime_bundle(tmp_path)
    (runtime_dir / "wheelhouse" / f"protcross-{stale}-py3-none-any.whl").write_bytes(b"stale wheel\n")

    result = subprocess.run(
        [
            sys.executable,
            "desktop/installer/validate_runtime_bundle.py",
            "--runtime-dir",
            str(runtime_dir),
            "--backend",
            "cpu",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "stale protcross wheel" in result.stderr
    assert f"expected version {expected}" in result.stderr


def test_release_version_consistency_validator_accepts_repository():
    result = subprocess.run(
        [sys.executable, "desktop/installer/validate_version_consistency.py", "--repo-root", "."],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"ProtCross {_core_version()}" in result.stdout


@pytest.mark.parametrize(
    ("surface", "expected_error"),
    (
        ("core", "protcross.__version__"),
        ("backend", "desktop backend project version"),
        ("backend_pin", "desktop backend protcross pin"),
        ("frontend", "frontend package version"),
        ("frontend_lock", "frontend lockfile version"),
        ("tauri_cargo", "Tauri Cargo package version"),
        ("tauri_lock", "Tauri Cargo lock package version"),
        ("tauri", "Tauri config version"),
        ("runtime", "desktop runtime protcross pin"),
        ("workflow", "desktop release workflow DESKTOP_VERSION"),
    ),
)
def test_release_version_consistency_validator_rejects_surface_drift(
    tmp_path: Path,
    surface: str,
    expected_error: str,
):
    fixture = _copy_version_fixture(tmp_path)
    current = _core_version()
    stale = "0.2.0" if current != "0.2.0" else "0.1.9"
    _make_version_surface_stale(fixture, surface, current=current, stale=stale)

    result = subprocess.run(
        [
            sys.executable,
            "desktop/installer/validate_version_consistency.py",
            "--repo-root",
            str(fixture),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert expected_error in result.stderr


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
    assert "validate_version_consistency.py" in release_inputs
    assert "cd frontend" in tauri_config["build"]["beforeBuildCommand"]
    assert "cd desktop/frontend" in tauri_config["build"]["beforeBuildCommand"]
    assert "cd ../frontend" in tauri_config["build"]["beforeBuildCommand"]
    assert tauri_config["bundle"]["category"] == "Education"
    assert "--backend cpu" in mac_sign_script
    assert "--backend cpu" in mac_validate_script
    assert "--backend all" not in mac_sign_script
    assert "--backend all" not in mac_validate_script


def test_readme_keeps_quick_start_short_and_desktop_details_in_installation():
    readme = Path("README.md").read_text(encoding="utf-8")
    quick_start = readme.split("## Quick start", 1)[1].split("## Contents", 1)[0]
    desktop = readme.split("## Desktop application", 1)[1].split(
        "## Training and development", 1
    )[0]

    assert "GitHub Releases" in quick_start
    assert "protcross inspect input.pdb" in quick_start
    assert "protcross setup-assets --accept-esm-license" in quick_start
    assert "protcross predict input.pdb" in quick_start
    assert "ProtCross_Desktop_0.2.2_x64-setup.exe" not in quick_start
    assert "ProtCross_Desktop_0.2.2_macos-aarch64.dmg" not in quick_start

    assert "ProtCross_Desktop_0.2.2_x64-setup.exe" in desktop
    assert "ProtCross_Desktop_0.2.2_macos-aarch64.dmg" in desktop
    assert "Tauri 2" in desktop
    assert "Mol*" in desktop


def test_packaged_sidecar_smoke_uses_env_token_only():
    validator = Path("desktop/installer/validate_packaged_sidecar.py").read_text(encoding="utf-8")
    tauri_main = Path("desktop/src-tauri/src/main.rs").read_text(encoding="utf-8")

    assert "PROTCROSS_DESKTOP_TOKEN" in validator
    assert "ProxyHandler({})" in validator
    assert 'default=60.0' in validator
    assert '"--token"' not in validator
    assert 'command.env("PROTCROSS_DESKTOP_TOKEN", &token)' in tauri_main
    assert '.arg("--token")' not in tauri_main


def test_desktop_routes_large_asset_downloads_through_background_jobs_only():
    server = Path("desktop/backend/protcross_desktop/server.py").read_text(encoding="utf-8")

    assert 'path == "/assets/download-esm/start"' in server
    assert 'path == "/assets/download-esm"' not in server


def test_gpu_online_install_reuses_bundled_release_wheelhouse():
    powershell = Path("desktop/runtime/install_gpu_backend.ps1").read_text(encoding="utf-8")
    posix = Path("desktop/runtime/install_gpu_backend.sh").read_text(encoding="utf-8")

    assert (
        'Invoke-Checked $Pip @("install", "--find-links", $Wheelhouse, "-r", $CommonRequirements)'
        in powershell
    )
    assert '"${PIP[@]}" install --find-links "$WHEELHOUSE" -r "$COMMON_REQUIREMENTS"' in posix


def test_desktop_release_workflow_locks_tag_and_limits_write_permission():
    workflow = Path(".github/workflows/desktop-release.yml").read_text(encoding="utf-8")
    publish_job = workflow.split("\n  publish:\n", 1)[1]

    assert re.search(r"(?m)^permissions:\n  contents: read$", workflow)
    assert "concurrency:\n  group: desktop-release-${{ inputs.tag }}\n  cancel-in-progress: false" in workflow
    assert re.search(r"(?m)^    permissions:\n      contents: write$", publish_job)
    assert workflow.count("fetch-depth: 0") == 4
    assert '"refs/heads/$DEFAULT_BRANCH"|"refs/tags/$RELEASE_TAG"' in workflow
    assert workflow.count('refs/tags/${RELEASE_TAG}^{commit}') >= 5
    assert "Verify Windows build commit and existing tag" in workflow
    assert "Verify macOS build commit and existing tag" in workflow
    assert "Verify publish commit and existing tag" in workflow
    assert "Verify published tag target" in workflow


def test_windows_release_scripts_validate_signed_runtime_and_installed_package():
    sign_script = Path("desktop/installer/sign-installer.ps1").read_text(encoding="utf-8")
    validate_script = Path("desktop/installer/validate-release.ps1").read_text(encoding="utf-8")
    tauri_config = json.loads(Path("desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    cpu_installer = Path("desktop/runtime/install_cpu_backend.ps1").read_text(encoding="utf-8")
    gpu_installer = Path("desktop/runtime/install_gpu_backend.ps1").read_text(encoding="utf-8")
    tauri_main = Path("desktop/src-tauri/src/main.rs").read_text(encoding="utf-8")

    assert "uv\\uv.exe" in sign_script
    assert "uv.sha256" in sign_script
    assert "Set-AuthenticodeSignature" in sign_script
    assert "ValidateInstalledPackage" in validate_script
    assert "RequireInstallerSignature" in validate_script
    assert "AcceptEsmLicenseForCi" not in validate_script
    assert "PROTCROSS_DESKTOP_CI_ACCEPT_ESMC_LICENSE" not in validate_script
    assert 'validate_runtime_bundle.py"), "--runtime-dir", $RuntimeDir, "--backend", "cpu"' in validate_script
    assert "validate_packaged_sidecar.py" in validate_script
    assert "install_cpu_backend.ps1" in validate_script
    assert "installerHooks" not in tauri_config["bundle"]["windows"]["nsis"]
    for script in (validate_script, cpu_installer, gpu_installer):
        assert "${LASTEXITCODE}:" in script
        assert "$LASTEXITCODE:" not in script
    assert '"-AllowOnlinePackageIndex"' in tauri_main
    assert '"--allow-online-package-index"' in tauri_main


def test_runtime_installers_seed_and_repair_pip_cross_platform():
    scripts = [
        Path("desktop/runtime/install_cpu_backend.ps1"),
        Path("desktop/runtime/install_gpu_backend.ps1"),
        Path("desktop/runtime/install_cpu_backend.sh"),
        Path("desktop/runtime/install_gpu_backend.sh"),
    ]

    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "--seed" in text, path
        assert "--allow-existing" in text, path
        assert "ensurepip" in text, path
        assert "Repair-EnvironmentPip" in text or "repair_environment_pip" in text, path


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_macos_runtime_installer_repairs_existing_environment_without_pip(tmp_path: Path, backend: str):
    desktop_dir = tmp_path / "desktop"
    runtime_dir = desktop_dir / "runtime"
    backend_dir = desktop_dir / "backend"
    runtime_dir.mkdir(parents=True)
    backend_dir.mkdir()
    script = runtime_dir / f"install_{backend}_backend.sh"
    shutil.copy2(Path("desktop/runtime") / script.name, script)
    script.chmod(0o755)
    (runtime_dir / "requirements-common.lock").write_text("example-package==1.0\n", encoding="utf-8")
    (runtime_dir / f"requirements-{backend}.hashes").write_text(
        "example-package==1.0 --hash=sha256:" + "0" * 64 + "\n",
        encoding="utf-8",
    )
    wheelhouse = runtime_dir / "wheelhouse"
    wheelhouse.mkdir()

    install_root = tmp_path / "install root"
    env_dir = install_root / "runtime" / f"{backend}-env"
    python = env_dir / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
marker="$(cd "$(dirname "$0")/.." && pwd)/.pip-ready"
if [[ "${1:-}" == "--version" ]]; then
  echo "Python 3.10.0"
  exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "ensurepip" ]]; then
  exit 1
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
  [[ -f "$marker" ]]
  exit
fi
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
fi
exit 0
""",
        encoding="utf-8",
    )
    python.chmod(0o755)

    uv_log = tmp_path / "uv-args.log"
    fake_uv = tmp_path / "fake-uv"
    fake_uv.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> {shlex.quote(str(uv_log))}
if [[ "${{1:-}}" == "venv" ]]; then
  env_dir="${{!#}}"
  touch "$env_dir/.pip-ready"
fi
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--install-root",
            str(install_root),
            "--python",
            "protcross-python-that-does-not-exist",
            "--uv-bin",
            str(fake_uv),
            "--wheelhouse",
            str(wheelhouse),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (env_dir / ".pip-ready").exists()
    uv_args = uv_log.read_text(encoding="utf-8")
    assert "venv --seed --allow-existing --python" in uv_args
    assert "missing pip" in result.stderr


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
    assert "--only-binary=:all:" in wheelhouse_script
    assert "local_desktop_backend_wheel" in wheelhouse_script
    assert "_remove_stale_release_wheels" in wheelhouse_script
    assert 'shutil.rmtree(package_dir / "build"' in wheelhouse_script
    assert 'features = ["protocol-asset"]' not in cargo_toml
    assert 'channel = "1.88.0"' in rust_toolchain
    assert "cargo +1.88.0 check --locked" in ci_workflow
    assert (
        "dtolnay/rust-toolchain@4e529fb27e59237866a6523e61ab248308c068b4 # 1.88.0"
        in ci_workflow
    )
    assert "--backend cpu" in ci_workflow
    assert "-AcceptEsmLicenseForCi" not in ci_workflow
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


def _make_runtime_bundle(root: Path, *, version: str | None = None) -> Path:
    version = version or _core_version()
    runtime_dir = root / "runtime"
    wheelhouse = runtime_dir / "wheelhouse"
    uv_dir = runtime_dir / "uv"
    wheelhouse.mkdir(parents=True)
    uv_dir.mkdir(parents=True)
    (runtime_dir / "requirements-common.lock").write_text(f"protcross=={version}\n", encoding="utf-8")
    wheels = {
        "protcross": wheelhouse / f"protcross-{version}-py3-none-any.whl",
        "protcross-desktop-backend": wheelhouse / f"protcross_desktop_backend-{version}-py3-none-any.whl",
        "torch": wheelhouse / "torch-2.3.1-cp310-cp310-win_amd64.whl",
    }
    for name, wheel in wheels.items():
        wheel.write_bytes(f"fake {name} wheel\n".encode("utf-8"))
    for backend in ("cpu", "gpu"):
        (runtime_dir / f"requirements-{backend}.hashes").write_text(
            "\n".join(
                [
                    f"protcross=={version} --hash=sha256:{_sha256(wheels['protcross'])}",
                    (
                        f"protcross-desktop-backend=={version} "
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


def _core_version() -> str:
    match = re.search(
        r'^version\s*=\s*"([^"]+)"\s*$',
        Path("pyproject.toml").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match
    return match.group(1)


def _copy_version_fixture(root: Path) -> Path:
    fixture = root / "version-fixture"
    relative_files = (
        "pyproject.toml",
        "src/protcross/__init__.py",
        "desktop/backend/pyproject.toml",
        "desktop/frontend/package.json",
        "desktop/frontend/package-lock.json",
        "desktop/src-tauri/Cargo.toml",
        "desktop/src-tauri/Cargo.lock",
        "desktop/src-tauri/tauri.conf.json",
        "desktop/runtime/requirements-common.lock",
        ".github/workflows/desktop-release.yml",
    )
    for relative in relative_files:
        source = Path(relative)
        target = fixture / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return fixture


def _make_version_surface_stale(root: Path, surface: str, *, current: str, stale: str) -> None:
    if surface == "core":
        paths = (root / "src/protcross/__init__.py",)
    elif surface == "backend":
        paths = (root / "desktop/backend/pyproject.toml",)
    elif surface == "backend_pin":
        path = root / "desktop/backend/pyproject.toml"
        text = path.read_text(encoding="utf-8")
        current_literal = f"protcross=={current}"
        assert current_literal in text
        path.write_text(text.replace(current_literal, f"protcross=={stale}", 1), encoding="utf-8")
        return
    elif surface == "frontend":
        paths = (root / "desktop/frontend/package.json",)
    elif surface == "frontend_lock":
        path = root / "desktop/frontend/package-lock.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["version"] = stale
        path.write_text(json.dumps(payload), encoding="utf-8")
        return
    elif surface == "tauri_cargo":
        paths = (root / "desktop/src-tauri/Cargo.toml",)
    elif surface == "tauri_lock":
        path = root / "desktop/src-tauri/Cargo.lock"
        text = path.read_text(encoding="utf-8")
        pattern = re.compile(
            rf'(\[\[package\]\]\nname = "protcross-desktop"\nversion = "){re.escape(current)}(")'
        )
        updated, count = pattern.subn(rf"\g<1>{stale}\g<2>", text, count=1)
        assert count == 1
        path.write_text(updated, encoding="utf-8")
        return
    elif surface == "tauri":
        paths = (root / "desktop/src-tauri/tauri.conf.json",)
    elif surface == "runtime":
        paths = (root / "desktop/runtime/requirements-common.lock",)
    elif surface == "workflow":
        path = root / ".github/workflows/desktop-release.yml"
        text = path.read_text(encoding="utf-8")
        current_literal = f'DESKTOP_VERSION: "{current}"'
        assert current_literal in text
        path.write_text(
            text.replace(current_literal, f'DESKTOP_VERSION: "{stale}"', 1),
            encoding="utf-8",
        )
        return
    else:
        raise AssertionError(f"unknown version surface: {surface}")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert current in text
        path.write_text(text.replace(current, stale, 1), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
