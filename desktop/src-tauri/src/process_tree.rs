//! Own desktop workers as process trees so an installer cannot outlive its data-root lease.

use std::io;
use std::process::{Child, Command, ExitStatus};

pub fn configure(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
}

pub fn terminate(child: &mut Child) -> io::Result<ExitStatus> {
    #[cfg(unix)]
    {
        extern "C" {
            fn kill(pid: i32, signal: i32) -> i32;
        }
        // Workers start in their own process group; pip, uv, and model workers inherit it.
        unsafe { kill(-(child.id() as i32), 9) };
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = Command::new("taskkill.exe")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .creation_flags(0x0800_0000)
            .output();
    }
    // Also cover a worker that exited before its tree was terminated, then reap it.
    let _ = child.kill();
    child.wait()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::path::Path;
    use std::process::Stdio;
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn delayed_descendant(marker: &Path) -> Command {
        #[cfg(unix)]
        {
            let mut command = Command::new("sh");
            command.args([
                "-c",
                "(sleep 1; printf orphan > \"$1\") & printf 'ready\\n'; wait",
                "protcross-test",
            ]);
            command.arg(marker);
            command
        }
        #[cfg(windows)]
        {
            let mut command = Command::new("powershell.exe");
            command.args(["-NoProfile", "-Command", "$child = Start-Process powershell.exe -PassThru -ArgumentList '-NoProfile', '-Command', ('Start-Sleep -Milliseconds 1000; Set-Content -LiteralPath ''{0}'' orphan' -f $env:PROTCROSS_TEST_MARKER); Write-Output ready; $child.WaitForExit()"]);
            command.env("PROTCROSS_TEST_MARKER", marker);
            command
        }
    }

    #[test]
    fn descendant_fixture_really_writes_without_cancellation() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "protcross-process-control-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let marker = root.join("completion-marker");
        let mut command = delayed_descendant(&marker);
        configure(&mut command);
        command.stdout(Stdio::null()).stderr(Stdio::null());
        assert!(command.status().unwrap().success());
        assert!(
            marker.exists(),
            "descendant fixture did not execute its delayed write"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn terminating_a_worker_stops_its_descendants_before_releasing_ownership() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "protcross-process-tree-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let marker = root.join("orphan-marker");
        let mut command = delayed_descendant(&marker);
        configure(&mut command);
        command.stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child = command.spawn().unwrap();
        let mut ready = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut ready)
            .unwrap();
        assert_eq!(ready.trim(), "ready");
        terminate(&mut child).unwrap();
        assert!(child.try_wait().unwrap().is_some());
        thread::sleep(Duration::from_millis(1400));
        assert!(
            !marker.exists(),
            "installer descendant survived and continued modifying files"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
