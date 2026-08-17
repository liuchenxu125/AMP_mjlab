#!/usr/bin/env python3
"""
Auto-resume training daemon for amp_mjlab.

Monitors the CASBOT02 leg AMP training process and automatically resumes
from the latest checkpoint when training dies (e.g., CUDA driver segfault).

Usage:
  # Fresh start
  python scripts/auto_resume.py Casbot02-Leg-AMP-Flat --max-iter 121001

  # Resume from existing training (finds latest log dir & checkpoint)
  python scripts/auto_resume.py Casbot02-Leg-AMP-Flat --max-iter 121001

  # Resume from a specific checkpoint
  python scripts/auto_resume.py Casbot02-Leg-AMP-Flat --max-iter 121001 \\
      --load-run 2026-08-10_09-42-44 --load-ckpt model_54000.pt

  # Custom check interval (seconds)
  python scripts/auto_resume.py Casbot02-Leg-AMP-Flat --check-interval 180
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "casbot02_leg_amp_locomotion"

CHECK_INTERVAL = 300       # seconds between health checks
STARTUP_WAIT = 15          # seconds after launch before scanning for log dir
HANG_TIMEOUT = 7200        # 2h without new checkpoint = hung
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def latest_log_dir() -> Optional[Path]:
    """Return the most recently modified log directory."""
    if not LOG_ROOT.is_dir():
        return None
    dirs = sorted(LOG_ROOT.glob("20*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        # Only consider directories that contain model checkpoints
        if list(d.glob("model_*.pt")):
            return d
    # Fallback: return newest directory even if empty
    return dirs[0] if dirs else None


def latest_checkpoint(log_dir: Path) -> tuple[Optional[Path], int]:
    """Return (path, iteration) of latest checkpoint in log_dir."""
    if not log_dir.is_dir():
        return None, 0
    models = sorted(log_dir.glob("model_*.pt"))
    if not models:
        return None, 0
    latest = models[-1]
    m = re.search(r"model_(\d+)\.pt", latest.name)
    iter_num = int(m.group(1)) if m else 0
    return latest, iter_num


def launch_training(
    task_name: str,
    max_iter: int,
    load_run: str | None = None,
    load_ckpt: str | None = None,
) -> tuple[subprocess.Popen, Path]:
    """Launch training subprocess. Returns (Popen, nohup_log_path)."""
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        task_name,
        f"--agent.max_iterations={max_iter}",
    ]
    if load_run and load_ckpt:
        cmd += [
            "--agent.resume=True",
            f"--agent.load_run={load_run}",
            f"--agent.load_checkpoint={load_ckpt}",
        ]

    label = "resume" if load_run else "init"
    ckpt_tag = f"_{re.sub(r'model_|\.pt', '', load_ckpt)}" if load_ckpt else ""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nohup_name = f"nohup_auto_resume_{label}{ckpt_tag}_{ts}.log"
    nohup_path = REPO_ROOT / nohup_name

    if load_run and load_ckpt:
        ckpt_iter = re.search(r"(\d+)", load_ckpt).group(1)
        log(f"[RESUME] from {load_run}/{load_ckpt} (iter={ckpt_iter})")
    else:
        log(f"[INIT] fresh start")

    log(f"  CMD: {' '.join(cmd)}")
    log(f"  NOHUP: {nohup_name}")

    with open(nohup_path, "w") as f:
        f.write(f"# Auto-resume {'resume' if load_run else 'init'} — {ts}\n")
        f.write(f"# TASK: {task_name}  MAX_ITER: {max_iter}\n")
        if load_run:
            f.write(f"# LOAD: {load_run}/{load_ckpt}\n")
        f.write(f"# CMD: {' '.join(cmd)}\n")
        f.write(f"{'=' * 80}\n\n")

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=open(nohup_path, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(f"  PID={proc.pid}")
    return proc, nohup_path


def discover_log_dir_or_die(oldest_known: Optional[Path]) -> Optional[Path]:
    """Find the log directory created by the most recent training launch."""
    time.sleep(STARTUP_WAIT)
    new_dir = latest_log_dir()
    if new_dir is None:
        log("WARNING: No log directory found yet")
        return None
    if oldest_known is None or new_dir != oldest_known:
        log(f"Tracking log dir: {new_dir.name}")
    return new_dir


def main():
    parser = argparse.ArgumentParser(
        description="Auto-resume training daemon for amp_mjlab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "task", nargs="?", default="Casbot02-Leg-AMP-Flat",
        help="Task ID to train (default: Casbot02-Leg-AMP-Flat)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=121001,
        help="Maximum training iterations (default: 121001)",
    )
    parser.add_argument(
        "--check-interval", type=int, default=CHECK_INTERVAL,
        help=f"Seconds between health checks (default: {CHECK_INTERVAL})",
    )
    parser.add_argument(
        "--hang-timeout", type=int, default=HANG_TIMEOUT,
        help=f"Seconds without new ckpt before declaring hung (default: {HANG_TIMEOUT})",
    )
    parser.add_argument(
        "--load-run", type=str, default=None,
        help="Resume from a specific log run directory",
    )
    parser.add_argument(
        "--load-ckpt", type=str, default=None,
        help="Resume from a specific checkpoint file (e.g., model_54000.pt)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Force fresh start, ignoring any existing checkpoints",
    )
    args = parser.parse_args()

    task_name: str = args.task
    max_iter: int = args.max_iter
    check_interval: int = args.check_interval
    hang_timeout: int = args.hang_timeout

    os.chdir(str(REPO_ROOT))

    log("=" * 60)
    log("Auto-Resume Training Daemon")
    log(f"  Task:       {task_name}")
    log(f"  Max iter:   {max_iter}")
    log(f"  Check:      every {check_interval}s")
    log(f"  Hang limit: {hang_timeout}s")
    log("=" * 60)

    # --- Determine initial resume state ---
    load_run: str | None = args.load_run
    load_ckpt: str | None = args.load_ckpt

    if args.fresh:
        log("--fresh: starting from scratch, ignoring existing checkpoints")
    elif load_run is None and load_ckpt is None:
        # Auto-detect: look for latest training to resume from
        latest_dir = latest_log_dir()
        if latest_dir is not None:
            ckpt_path, ckpt_iter = latest_checkpoint(latest_dir)
            if ckpt_iter >= max_iter:
                log(f"Training already complete: {latest_dir.name} "
                    f"iter={ckpt_iter} >= {max_iter}")
                return
            if ckpt_path is not None:
                log(f"Auto-detected: {latest_dir.name} / {ckpt_path.name} (iter={ckpt_iter})")
                # Only auto-resume if last activity was recent (within 24h)
                age_s = time.time() - ckpt_path.stat().st_mtime
                if age_s < 86400:
                    load_run = latest_dir.name
                    load_ckpt = ckpt_path.name
                    log(f"  Will resume (ckpt is {age_s/3600:.1f}h old, within 24h)")
                else:
                    log(f"  Skipping resume (ckpt is {age_s/3600:.1f}h old, >24h)")
                    log(f"  Starting fresh. Use --load-run to force resume.")

    # --- Launch initial training ---
    current_proc, current_nohup = launch_training(
        task_name, max_iter, load_run, load_ckpt,
    )
    resume_count = 0
    current_log_dir = discover_log_dir_or_die(None)

    # --- Monitor loop ---
    def handle_exit():
        nonlocal current_proc, current_log_dir, current_nohup, resume_count

        # Process died — find latest checkpoint
        log_dir = current_log_dir or latest_log_dir()
        if log_dir is None:
            log("FATAL: No log directory found, cannot resume")
            return False

        ckpt_path, ckpt_iter = latest_checkpoint(log_dir)
        if ckpt_iter >= max_iter:
            log(f"✅ Training complete! iter={ckpt_iter} >= {max_iter}")
            return False

        if ckpt_path is None:
            log("FATAL: Process died with no checkpoint, cannot resume")
            return False

        # Check if the last checkpoint is very recent (within 5 min)
        # If so, the training may have crashed quickly — apply backoff
        age_s = time.time() - ckpt_path.stat().st_mtime
        if age_s < 300 and resume_count > 0:
            wait = min(60 * (2 ** (resume_count - 1)), 1800)  # max 30min backoff
            log(f"Rapid re-crash detected (ckpt {age_s:.0f}s old), "
                f"backing off {wait}s...")
            time.sleep(wait)

        resume_count += 1
        log(f"Auto-resume #{resume_count}: {log_dir.name}/{ckpt_path.name} "
            f"(iter={ckpt_iter})")

        current_proc, current_nohup = launch_training(
            task_name, max_iter,
            load_run=log_dir.name,
            load_ckpt=ckpt_path.name,
        )
        current_log_dir = discover_log_dir_or_die(log_dir)
        return True

    try:
        while True:
            time.sleep(check_interval)

            # --- Health check 1: is the process alive? ---
            exit_code = current_proc.poll()
            if exit_code is not None:
                log(f"💀 Process PID={current_proc.pid} died (exit code={exit_code})")
                if not handle_exit():
                    break
                continue

            # --- Health check 2: is training making progress? ---
            log_dir = current_log_dir
            if log_dir is None:
                log_dir = latest_log_dir()
                if log_dir is not None:
                    current_log_dir = log_dir

            if log_dir is not None:
                ckpt_path, ckpt_iter = latest_checkpoint(log_dir)
                if ckpt_path is not None:
                    ago_s = time.time() - ckpt_path.stat().st_mtime

                    # Build a compact status line
                    parts = [f"PID={current_proc.pid}", f"iter={ckpt_iter}"]
                    if ago_s < 3600:
                        parts.append(f"ckpt {ago_s/60:.0f}m ago")
                    else:
                        parts.append(f"ckpt {ago_s/3600:.1f}h ago")
                    if resume_count > 0:
                        parts.append(f"resumes={resume_count}")
                    log(" | ".join(parts))

                    # Hang detection
                    if ago_s > hang_timeout:
                        log(f"🧟 HANG DETECTED: no new ckpt for {ago_s/3600:.1f}h "
                            f"(limit={hang_timeout/3600:.1f}h)")
                        log(f"  Killing PID={current_proc.pid}...")
                        try:
                            os.killpg(os.getpgid(current_proc.pid), signal.SIGTERM)
                        except (ProcessLookupError, OSError):
                            pass
                        try:
                            current_proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            log("  Force killing...")
                            current_proc.kill()
                            current_proc.wait(timeout=10)
                        if not handle_exit():
                            break
                        continue
                else:
                    log(f"PID={current_proc.pid} alive | no ckpt yet")
            else:
                log(f"PID={current_proc.pid} alive")

    except KeyboardInterrupt:
        log("Received SIGINT, shutting down...")
        log("Training continues in background (start_new_session).")
        log(f"To re-attach, run with: --load-run {current_log_dir.name if current_log_dir else '<dir>'} "
            f"--load-ckpt <latest>.pt")

    log("Auto-resume daemon exiting.")


if __name__ == "__main__":
    main()
