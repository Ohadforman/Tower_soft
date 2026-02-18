import os
import platform
import subprocess
from datetime import datetime


def run_after_done_hook(
    target_csv: str,
    done_desc: str,
    preform_len_after_cm: float,
    hook_dir: str = "hooks",
    timeout_sec: int = 120,
):
    """
    Runs an OS-specific 'after done' hook:
      - mac/linux: hooks/after_done.sh
      - windows:   hooks/after_done.bat

    Always writes a text log:
      hooks/after_done_last_run.txt

    Returns (ok: bool, msg: str).
    """
    os.makedirs(hook_dir, exist_ok=True)

    # ---- log file (always)
    log_path = os.path.join(hook_dir, "after_done_last_run.txt")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- sanitize description for passing to scripts (avoid quotes/newlines issues)
    safe_desc = (done_desc or "").replace("\r", " ").replace("\n", " ").strip()
    safe_desc = safe_desc[:500]  # keep it short for cmdline safety

    # ---- prepare args
    args = [
        str(target_csv),
        f"{float(preform_len_after_cm):.1f}",
        safe_desc,
    ]

    # ---- decide OS + script path
    is_windows = (os.name == "nt") or (platform.system().lower().startswith("win"))
    if is_windows:
        script_path = os.path.join(hook_dir, "after_done.bat")
        cmd = [script_path] + args
    else:
        script_path = os.path.join(hook_dir, "after_done.sh")
        cmd = ["bash", script_path] + args

    # ---- write a header to log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== AFTER DONE HOOK RUN ===\n")
        f.write(f"Time: {now_str}\n")
        f.write(f"OS: {platform.system()} ({os.name})\n")
        f.write(f"Script: {os.path.abspath(script_path)}\n")
        f.write(f"Args: {args}\n")
        f.write("\n")

    if not os.path.exists(script_path):
        msg = (
            f"Hook script not found: {script_path}\n"
            f"Create it to enable after-done actions.\n"
            f"Log: {log_path}"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        return False, msg

    # ---- run
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )

        stdout = (p.stdout or "").strip()
        stderr = (p.stderr or "").strip()

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Exit code: {p.returncode}\n\n")
            if stdout:
                f.write("---- STDOUT ----\n")
                f.write(stdout + "\n\n")
            if stderr:
                f.write("---- STDERR ----\n")
                f.write(stderr + "\n\n")

        if p.returncode == 0:
            return True, f"After-done hook OK. Log written to: {log_path}"
        return False, f"After-done hook FAILED (exit {p.returncode}). Log: {log_path}"

    except subprocess.TimeoutExpired:
        msg = f"After-done hook timed out after {timeout_sec}s. Log: {log_path}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        return False, msg

    except Exception as e:
        msg = f"Failed running after-done hook: {e}. Log: {log_path}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        return False, msg