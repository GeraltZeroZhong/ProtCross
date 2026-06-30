use std::fs::{File, OpenOptions};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use tauri::{AppHandle, Manager, State};

struct BackendProcess {
    child: Mutex<Option<Child>>,
    token: Mutex<Option<String>>,
    port: Mutex<Option<u16>>,
}

#[derive(serde::Serialize)]
struct BackendStartResult {
    token: String,
    port: u16,
}

impl Drop for BackendProcess {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

#[tauri::command]
fn start_backend(
    app: AppHandle,
    state: State<BackendProcess>,
    port: Option<u16>,
    root: Option<String>,
) -> Result<BackendStartResult, String> {
    let mut guard = state.child.lock().map_err(|_| "backend lock poisoned".to_string())?;
    let mut token_guard = state.token.lock().map_err(|_| "backend token lock poisoned".to_string())?;
    let mut port_guard = state.port.lock().map_err(|_| "backend port lock poisoned".to_string())?;
    if let Some(child) = guard.as_mut() {
        match child.try_wait() {
            Ok(Some(_status)) => {
                *guard = None;
                *token_guard = None;
                *port_guard = None;
            }
            Ok(None) => {
                if let (Some(token), Some(port)) = (token_guard.as_ref(), *port_guard) {
                    return Ok(BackendStartResult {
                        token: token.clone(),
                        port,
                    });
                }
                return Err("backend is already running but no API token is available; restart it".to_string());
            }
            Err(_exc) => {
                *guard = None;
                *token_guard = None;
                *port_guard = None;
            }
        }
    }

    let root_path = root.as_deref().map(PathBuf::from).unwrap_or_else(|| app_data_root(&app));
    let log_dir = root_path.join("logs");
    std::fs::create_dir_all(&log_dir).map_err(|exc| format!("failed to create backend log dir: {exc}"))?;
    let log_file = open_log_file(&log_dir.join("backend.log"))?;
    let stderr_file = log_file.try_clone().map_err(|exc| format!("failed to clone backend log: {exc}"))?;

    let resource_dir = app.path().resource_dir().ok();
    let backend_paths = backend_python_paths(resource_dir.as_deref());
    let bundled_assets = std::env::var("PROTCROSS_DESKTOP_BUNDLED_ASSETS")
        .ok()
        .or_else(|| resource_dir.as_ref().map(|path| path.join("bundled-assets").to_string_lossy().to_string()));
    let python = std::env::var("PROTCROSS_DESKTOP_PYTHON")
        .unwrap_or_else(|_| configured_python(Some(root_path.as_path())).unwrap_or_else(|| "python".to_string()));
    let token = generate_token()?;
    let selected_port = match port {
        Some(value) if value != 0 => value,
        _ => find_available_port()?,
    };

    let mut command = Command::new(python);
    if !backend_paths.is_empty() {
        let mut paths = backend_paths;
        if let Some(existing) = std::env::var_os("PYTHONPATH") {
            paths.extend(std::env::split_paths(&existing));
        }
        let joined = std::env::join_paths(paths).map_err(|exc| format!("invalid PYTHONPATH: {exc}"))?;
        command.env("PYTHONPATH", joined);
    }
    if let Some(path) = bundled_assets {
        command.env("PROTCROSS_DESKTOP_BUNDLED_ASSETS", path);
    }
    command.env("PROTCROSS_DESKTOP_TOKEN", &token);
    command
        .arg("-m")
        .arg("protcross_desktop.server")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(selected_port.to_string())
        .arg("--root")
        .arg(&root_path)
        .arg("--token")
        .arg(&token)
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(stderr_file));
    let mut child = command.spawn().map_err(|exc| format!("failed to start backend: {exc}"))?;
    thread::sleep(Duration::from_millis(300));
    if let Ok(Some(status)) = child.try_wait() {
        return Err(format!(
            "backend exited immediately with {status}; see {}",
            log_dir.join("backend.log").display()
        ));
    }
    *guard = Some(child);
    *token_guard = Some(token.clone());
    *port_guard = Some(selected_port);
    Ok(BackendStartResult {
        token,
        port: selected_port,
    })
}

#[tauri::command]
fn stop_backend(state: State<BackendProcess>) -> Result<String, String> {
    let mut guard = state.child.lock().map_err(|_| "backend lock poisoned".to_string())?;
    let mut token_guard = state.token.lock().map_err(|_| "backend token lock poisoned".to_string())?;
    let mut port_guard = state.port.lock().map_err(|_| "backend port lock poisoned".to_string())?;
    if let Some(mut child) = guard.take() {
        let _ = child.kill();
        let _ = child.wait();
        *token_guard = None;
        *port_guard = None;
        return Ok("stopped".to_string());
    }
    *token_guard = None;
    *port_guard = None;
    Ok("not running".to_string())
}

#[tauri::command]
fn install_backend(
    app: AppHandle,
    mode: String,
    root: Option<String>,
    proxy_url: Option<String>,
) -> Result<String, String> {
    if mode != "cpu" && mode != "gpu" {
        return Err("backend mode must be cpu or gpu".to_string());
    }
    let root_path = root.as_deref().map(PathBuf::from).unwrap_or_else(|| app_data_root(&app));
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|exc| format!("failed to resolve app resources: {exc}"))?;
    let runtime_dir = resource_dir.join("runtime");
    let logs_dir = root_path.join("logs");
    std::fs::create_dir_all(&logs_dir).map_err(|exc| format!("failed to create log dir: {exc}"))?;
    let log_path = logs_dir.join(format!("runtime-install-{mode}.log"));
    let stdout = open_log_file(&log_path)?;
    let stderr = stdout.try_clone().map_err(|exc| format!("failed to clone install log: {exc}"))?;

    let mut command = runtime_install_command(&runtime_dir, &mode, &root_path, proxy_url.as_deref())?;
    let status = command
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .status()
        .map_err(|exc| format!("failed to run backend installer: {exc}"))?;
    if !status.success() {
        return Err(format!(
            "{mode} backend installer failed with {status}; see {}",
            log_path.display()
        ));
    }
    Ok(format!("{mode} backend installed; log: {}", log_path.display()))
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    let path = PathBuf::from(path);
    if !path.exists() {
        return Err("path does not exist".to_string());
    }
    #[cfg(target_os = "windows")]
    {
        Command::new("explorer")
            .arg(path)
            .spawn()
            .map_err(|exc| format!("failed to open path: {exc}"))?;
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(path)
            .spawn()
            .map_err(|exc| format!("failed to open path: {exc}"))?;
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open")
            .arg(path)
            .spawn()
            .map_err(|exc| format!("failed to open path: {exc}"))?;
    }
    Ok(())
}

#[tauri::command]
fn open_url(url: String) -> Result<(), String> {
    if !is_trusted_url(&url) {
        return Err("refusing to open an untrusted URL".to_string());
    }
    #[cfg(target_os = "windows")]
    {
        Command::new("explorer.exe")
            .arg(url)
            .spawn()
            .map_err(|exc| format!("failed to open URL: {exc}"))?;
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(url)
            .spawn()
            .map_err(|exc| format!("failed to open URL: {exc}"))?;
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open")
            .arg(url)
            .spawn()
            .map_err(|exc| format!("failed to open URL: {exc}"))?;
    }
    Ok(())
}

fn is_trusted_url(url: &str) -> bool {
    if url.chars().any(|c| {
        c.is_control() || matches!(c, '"' | '\'' | '`' | '&' | '|' | ';' | '<' | '>' | '^')
    }) {
        return false;
    }
    url == "https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement"
        || url.starts_with("https://github.com/GeraltZeroZhong/ProtCross/")
}

fn configured_python(root: Option<&Path>) -> Option<String> {
    let root = root.map(PathBuf::from).unwrap_or_else(default_desktop_root);
    let manifest = root.join("assets").join("protcross-desktop-assets.json");
    let value: Option<serde_json::Value> = std::fs::read_to_string(manifest)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok());
    let Some(payload) = value else {
        return first_existing_managed_python(&root);
    };
    let Some(mode) = payload.get("backend_mode").and_then(|mode| mode.as_str()) else {
        return first_existing_managed_python(&root);
    };
    let python = match mode {
        "cpu" | "gpu" => root
            .join("runtime")
            .join(format!("{mode}-env"))
            .join(env_python_relative()),
        "conda" => PathBuf::from(payload.get("conda_python")?.as_str()?),
        _ => return None,
    };
    if python.exists() {
        Some(python.to_string_lossy().to_string())
    } else {
        None
    }
}

fn first_existing_managed_python(root: &Path) -> Option<String> {
    for mode in ["gpu", "cpu"] {
        let python = root
            .join("runtime")
            .join(format!("{mode}-env"))
            .join(env_python_relative());
        if python.exists() {
            return Some(python.to_string_lossy().to_string());
        }
    }
    None
}

fn runtime_install_command(
    runtime_dir: &Path,
    mode: &str,
    root: &Path,
    proxy_url: Option<&str>,
) -> Result<Command, String> {
    #[cfg(target_os = "windows")]
    {
        let script = runtime_dir.join(format!("install_{mode}_backend.ps1"));
        if !script.exists() {
            return Err(format!("backend installer not found: {}", script.display()));
        }
        let mut command = Command::new("powershell.exe");
        command
            .arg("-NoProfile")
            .arg("-ExecutionPolicy")
            .arg("Bypass")
            .arg("-File")
            .arg(script)
            .arg("-InstallRoot")
            .arg(root);
        if let Some(proxy) = proxy_url.filter(|value| !value.is_empty()) {
            command.arg("-ProxyUrl").arg(proxy);
        }
        command.arg("-Wheelhouse").arg(runtime_dir.join("wheelhouse"));
        return Ok(command);
    }
    #[cfg(not(target_os = "windows"))]
    {
        let script = runtime_dir.join(format!("install_{mode}_backend.sh"));
        if !script.exists() {
            return Err(format!("backend installer not found: {}", script.display()));
        }
        let mut command = Command::new("bash");
        command.arg(script).arg("--install-root").arg(root);
        if let Some(proxy) = proxy_url.filter(|value| !value.is_empty()) {
            command.arg("--proxy-url").arg(proxy);
        }
        command.arg("--wheelhouse").arg(runtime_dir.join("wheelhouse"));
        Ok(command)
    }
}

fn backend_python_paths(resource_dir: Option<&Path>) -> Vec<PathBuf> {
    if let Ok(path) = std::env::var("PROTCROSS_DESKTOP_BACKEND_PATH") {
        return vec![PathBuf::from(path)];
    }
    let Some(resource_dir) = resource_dir else {
        return Vec::new();
    };
    vec![resource_dir.join("backend"), resource_dir.join("python-src")]
}

fn open_log_file(path: &Path) -> Result<File, String> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|exc| format!("failed to open log file {}: {exc}", path.display()))
}

fn generate_token() -> Result<String, String> {
    let mut bytes = [0_u8; 32];
    getrandom::getrandom(&mut bytes).map_err(|exc| format!("failed to generate desktop API token: {exc}"))?;
    let mut token = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        token.push_str(&format!("{byte:02x}"));
    }
    Ok(token)
}

fn find_available_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|exc| format!("failed to allocate localhost backend port: {exc}"))?;
    let port = listener
        .local_addr()
        .map_err(|exc| format!("failed to read allocated backend port: {exc}"))?
        .port();
    drop(listener);
    Ok(port)
}

fn app_data_root(app: &AppHandle) -> PathBuf {
    if let Ok(root) = std::env::var("PROTCROSS_DESKTOP_HOME") {
        return PathBuf::from(root);
    }
    app.path().app_local_data_dir().unwrap_or_else(|_| default_desktop_root())
}

fn env_python_relative() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "Scripts/python.exe"
    }
    #[cfg(not(target_os = "windows"))]
    {
        "bin/python"
    }
}

fn default_desktop_root() -> PathBuf {
    if let Ok(root) = std::env::var("PROTCROSS_DESKTOP_HOME") {
        return PathBuf::from(root);
    }
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::var("USERPROFILE").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(".")));
        return base.join("ProtCross");
    }
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."));
        return home.join("Library").join("Application Support").join("ProtCross");
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let home = std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."));
        home.join(".local").join("share").join("protcross-desktop")
    }
}

fn main() {
    tauri::Builder::default()
        .manage(BackendProcess {
            child: Mutex::new(None),
            token: Mutex::new(None),
            port: Mutex::new(None),
        })
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            start_backend,
            stop_backend,
            install_backend,
            open_path,
            open_url
        ])
        .run(tauri::generate_context!())
        .expect("error while running ProtCross Desktop");
}
