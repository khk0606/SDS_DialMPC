from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _cmd(script_path: str, config: Path, custom_env: str | None) -> List[str]:
    cmd = [sys.executable, script_path, "--config", str(config)]
    if custom_env:
        cmd += ["--custom-env", custom_env]
    return cmd


def _terminate_process(proc: subprocess.Popen, grace_sec: float, sigint_first: bool = False) -> int:
    if proc.poll() is not None:
        return int(proc.returncode)
    if sigint_first:
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            proc.terminate()
    else:
        proc.terminate()
    try:
        proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch dial_sim and dial_plan together, stop plan when sim exits."
    )
    parser.add_argument("--config", required=True, help="Path to merged config yaml")
    parser.add_argument(
        "--custom-env",
        default="dial_mpc.envs.unitree_go2_env",
        help="Custom env module for both processes",
    )
    parser.add_argument(
        "--sim-script",
        default="dial-mpc/dial_mpc/deploy/dial_sim.py",
        help="Simulator script path",
    )
    parser.add_argument(
        "--plan-script",
        default="dial-mpc/dial_mpc/deploy/dial_plan.py",
        help="Planner script path",
    )
    parser.add_argument("--startup-delay-sec", type=float, default=1.5, help="Delay before starting planner")
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=0.0,
        help="Optional global timeout. 0 disables timeout.",
    )
    parser.add_argument("--plan-grace-sec", type=float, default=5.0, help="Graceful termination wait for planner")
    parser.add_argument("--workdir", default=".", help="Process working directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    config_path = Path(args.config).resolve()

    sim_cmd = _cmd(args.sim_script, config_path, args.custom_env)
    plan_cmd = _cmd(args.plan_script, config_path, args.custom_env)

    env = os.environ.copy()
    repo_pkg_path = str((workdir / "dial-mpc").resolve())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_pkg_path if not existing_pythonpath else f"{repo_pkg_path}:{existing_pythonpath}"

    print(f"[pair] launching sim: {' '.join(sim_cmd)}")
    sim_proc = subprocess.Popen(sim_cmd, cwd=str(workdir), env=env)
    plan_proc: subprocess.Popen | None = None
    timed_out = False
    plan_failed_rc: int | None = None

    try:
        time.sleep(max(0.0, args.startup_delay_sec))
        if sim_proc.poll() is None:
            print(f"[pair] launching plan: {' '.join(plan_cmd)}")
            plan_proc = subprocess.Popen(plan_cmd, cwd=str(workdir), env=env)
        else:
            print("[pair] sim exited before planner startup; skipping planner launch")

        t0 = time.time()
        while True:
            sim_rc = sim_proc.poll()
            if sim_rc is not None:
                break
            if plan_proc is not None:
                rc = plan_proc.poll()
                if rc is not None:
                    if rc != 0:
                        plan_failed_rc = int(rc)
                        print(f"[pair] planner exited with rc={rc}; terminating simulator")
                        _terminate_process(sim_proc, grace_sec=2.0)
                        break
                    else:
                        print("[pair] planner exited cleanly before simulator; terminating simulator")
                        _terminate_process(sim_proc, grace_sec=2.0)
                        break
            if args.timeout_sec > 0 and (time.time() - t0) > args.timeout_sec:
                timed_out = True
                print(f"[pair] timeout reached ({args.timeout_sec}s), terminating processes")
                _terminate_process(sim_proc, grace_sec=2.0)
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[pair] interrupted by user")
    finally:
        sim_rc = sim_proc.poll()
        if sim_rc is None:
            sim_rc = _terminate_process(sim_proc, grace_sec=2.0)

        plan_rc = 0
        if plan_proc is not None:
            plan_rc = _terminate_process(plan_proc, grace_sec=args.plan_grace_sec, sigint_first=True)

        print(f"[pair] sim_rc={sim_rc} plan_rc={plan_rc} timed_out={timed_out}")
        if timed_out:
            raise SystemExit(124)
        if plan_failed_rc is not None:
            raise SystemExit(plan_failed_rc)
        if int(sim_rc) != 0:
            raise SystemExit(int(sim_rc))


if __name__ == "__main__":
    main()
