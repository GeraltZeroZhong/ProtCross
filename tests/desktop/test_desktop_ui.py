from __future__ import annotations

import json
from pathlib import Path


FRONTEND = Path("desktop/frontend/src")


def test_desktop_design_system_supports_platform_appearance_and_accessibility():
    styles = (FRONTEND / "styles.css").read_text(encoding="utf-8")

    assert "oklch(" in styles
    assert ':root[data-theme="dark"]' in styles
    assert "prefers-color-scheme: dark" in styles
    assert "prefers-reduced-motion: reduce" in styles
    assert "prefers-contrast: more" in styles
    assert "forced-colors: active" in styles
    assert "button:focus-visible" in styles
    assert "select:focus-visible" in styles
    assert "min-height: 36px" in styles


def test_desktop_shell_exposes_navigation_and_async_state_semantics():
    app = (FRONTEND / "App.tsx").read_text(encoding="utf-8")

    assert 'aria-label="Primary navigation"' in app
    assert 'aria-current={selected ? "page" : undefined}' in app
    assert 'role="status"' in app
    assert 'role="alert"' in app
    assert 'className="skip-link"' in app
    assert 'id="workspace-content"' in app
    assert '<caption className="sr-only">' in app
    assert 'scope="col"' in app
    assert "const processed = props.batchJob?.completed ?? 0" in app
    assert "processed - (props.batchJob?.failed ?? 0)" in app
    assert "completed ?? 0) +" not in app
    assert "backendIsHealthy(props.status)" in app
    assert 'props.status?.readiness?.ready === true' in app


def test_result_view_uses_protcross_score_theme_and_legend():
    viewer = (FRONTEND / "components" / "MolstarViewer.tsx").read_text(encoding="utf-8")
    theme = (FRONTEND / "components" / "ProtcrossScoreTheme.ts").read_text(encoding="utf-8")

    assert "ProtcrossScoreColorThemeProvider" in viewer
    assert 'color: "protcross-score"' in viewer
    assert 'className="score-legend"' in viewer
    assert "B_iso_or_equiv.value(element)" in theme
    assert "domain: [0, 1]" in theme


def test_desktop_window_supports_compact_resizable_workspaces():
    config = json.loads(Path("desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    window = config["app"]["windows"][0]

    assert window["minWidth"] <= 860
    assert window["minHeight"] <= 680
    assert window["width"] >= 1400


def test_frontend_declares_browser_accessibility_regression_suite():
    package = json.loads(Path("desktop/frontend/package.json").read_text(encoding="utf-8"))
    spec = Path("desktop/frontend/e2e/ui.spec.ts").read_text(encoding="utf-8")

    assert package["scripts"]["test:ui"] == "playwright test"
    assert "@axe-core/playwright" in package["devDependencies"]
    assert "workspace reflows to 320 CSS pixels" in spec
    assert "visible pointer targets meet the WCAG minimum" in spec


def test_desktop_polling_recovers_from_backend_loss_without_permanent_active_state():
    app = (FRONTEND / "App.tsx").read_text(encoding="utf-8")
    api = (FRONTEND / "api.ts").read_text(encoding="utf-8")

    assert "withRequestDeadline" in app
    assert "consecutiveFailures >= 3" in app
    assert 'status: "interrupted"' in app
    assert "backendConnectionLost" in app
    assert "signal?: AbortSignal" in api


def test_tauri_security_disables_unused_asset_protocol_and_sets_strict_csp():
    config = json.loads(Path("desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    cargo = Path("desktop/src-tauri/Cargo.toml").read_text(encoding="utf-8")
    security = config["app"]["security"]

    assert "assetProtocol" not in security
    assert "protocol-asset" not in cargo
    assert "default-src 'self'" in security["csp"]
    assert "object-src 'none'" in security["csp"]
    assert "script-src 'self' 'wasm-unsafe-eval' 'unsafe-eval'" in security["csp"]
    assert "connect-src 'self' data: blob:" in security["csp"]
    assert "http://127.0.0.1:*" in security["csp"]
    assert "ws://127.0.0.1:5173" in security["devCsp"]


def test_tauri_runtime_commands_share_a_cross_process_root_lease():
    source = Path("desktop/src-tauri/src/main.rs").read_text(encoding="utf-8")

    assert "ensure_root_lease(&state, &root_path)?" in source
    assert source.count("ensure_root_lease(&state, &root_path)?") == 2
    assert '.join(".protcross-desktop.lock")' in source
    assert "LOCK_EX | LOCK_NB" in source
    assert ".share_mode(0)" in source
