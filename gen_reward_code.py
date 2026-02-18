from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests


@dataclass
class MotionSpec:
    mode: str                 # sit_stand | slow_walk | walk | trot_pace
    gait: str
    target_speed_mps: float
    base_height_m: float
    notes: str = ""


def parse_speed_mps(text: str) -> Optional[float]:
    t = text.lower()

    # 1) "0.3 - 0.6 m/s"
    m = re.search(r'(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*m/s', t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (a + b) * 0.5

    # 2) "4 - 7 km/h"
    m = re.search(r'(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*km/h', t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return ((a + b) * 0.5) / 3.6

    # 3) "1.1 m/s"
    m = re.search(r'(\d+(?:\.\d+)?)\s*m/s', t)
    if m:
        return float(m.group(1))

    # 4) "7 km/h"
    m = re.search(r'(\d+(?:\.\d+)?)\s*km/h', t)
    if m:
        return float(m.group(1)) / 3.6

    return None


def heuristic_spec(report: str) -> MotionSpec:
    t = report.lower()
    speed = parse_speed_mps(t)
    if speed is None:
        speed = 0.4

    gait = "walk"
    if "pace" in t or "pacing" in t:
        gait = "pace"
    elif "trot" in t:
        gait = "trot"
    elif "sit" in t or "sitting" in t or "stand still" in t:
        gait = "sit"

    if gait == "sit":
        mode = "sit_stand"
        target_speed = 0.0
    elif gait in ("pace", "trot") or speed >= 1.2:
        mode = "trot_pace"
        target_speed = max(speed, 1.0)
    elif speed <= 0.7:
        mode = "slow_walk"
        target_speed = max(speed, 0.25)
    else:
        mode = "walk"
        target_speed = max(speed, 0.5)

    # SUS에서 흔한 값이 0.30m라 기본 유지
    base_h = 0.30

    return MotionSpec(
        mode=mode,
        gait=gait,
        target_speed_mps=round(target_speed, 3),
        base_height_m=base_h,
        notes="heuristic spec",
    )


def _extract_json_block(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    m = re.search(r'\{.*\}', s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def gemini_spec(report: str, model: str, api_key: str) -> Optional[MotionSpec]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    prompt = f"""
You are extracting motion-control metadata for quadruped reward generation.
Return JSON only.

Allowed mode values: "sit_stand", "slow_walk", "walk", "trot_pace"
JSON schema:
{{
  "mode": str,
  "gait": str,
  "target_speed_mps": number,
  "base_height_m": number,
  "notes": str
}}

Rules:
- If report says sit/stand still, mode=sit_stand and target_speed_mps=0.0
- Slow walk should be around 0.25~0.7 m/s
- base_height_m in range 0.26~0.34 for Go2
- Output ONLY JSON

Report:
{report}
"""

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512},
    }

    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        txt = data["candidates"][0]["content"]["parts"][0]["text"]
        obj = _extract_json_block(txt)
        if not obj:
            return None

        mode = str(obj.get("mode", "slow_walk"))
        if mode not in {"sit_stand", "slow_walk", "walk", "trot_pace"}:
            mode = "slow_walk"

        gait = str(obj.get("gait", "walk"))
        v = float(obj.get("target_speed_mps", 0.4))
        h = float(obj.get("base_height_m", 0.30))
        notes = str(obj.get("notes", "gemini spec"))

        # clamp
        v = max(0.0, min(v, 3.0))
        h = max(0.26, min(h, 0.34))

        if mode == "sit_stand":
            v = 0.0

        return MotionSpec(mode=mode, gait=gait, target_speed_mps=round(v, 3), base_height_m=round(h, 3), notes=notes)
    except Exception:
        return None


def build_reward_code(spec: MotionSpec) -> str:
    # 공통 terms
    terms = [
        "target_vx = jnp.where(jnp.abs(vel_tar[0]) < 1e-6, "
        + f"{spec.target_speed_mps:.3f}"
        + ", vel_tar[0])",
        "reward_vel_x = -jnp.square(vb[0] - target_vx)",
        "penalty_orientation = jnp.square(roll) + jnp.square(pitch)",
        f"penalty_height = jnp.square(torso_pos[2] - {spec.base_height_m:.3f})",
        "penalty_vertical_vel = jnp.square(vb[2])",
        "penalty_accel = jnp.sum(jnp.square(ab))",
        "penalty_joint_pose = jnp.sum(jnp.square(joint_angles - default_pose))",
        "penalty_joint_vel = jnp.sum(jnp.square(joint_vel))",
        "penalty_energy = jnp.sum(jnp.square(ctrl))",
        "penalty_collapse = jnp.square(jnp.clip(0.26 - torso_pos[2], 0.0, 1.0))",
    ]

    weights = {
        "reward_vel_x": 8.0,
        "penalty_orientation": 8.0,
        "penalty_height": 6.0,
        "penalty_vertical_vel": 2.0,
        "penalty_accel": 0.06,
        "penalty_joint_pose": 0.12,
        "penalty_joint_vel": 0.001,
        "penalty_energy": 0.0003,
        "penalty_collapse": 12.0,
    }

    if spec.mode == "sit_stand":
        terms += [
            "reward_still = -(jnp.square(vb[0]) + jnp.square(vb[1]) + 0.5 * jnp.square(vb[2]))",
            "penalty_lateral_vel = jnp.square(vb[1])",
            "penalty_yaw_rate = jnp.square(ab[2] - ang_vel_tar[2])",
            "penalty_feet_too_high = jnp.sum(jnp.square(jnp.clip(feet_z - 0.08, 0.0, 1.0)))",
        ]
        weights.update({
            "reward_still": 8.0,
            "penalty_lateral_vel": 2.0,
            "penalty_yaw_rate": 1.0,
            "penalty_feet_too_high": 2.0,
            # sit/stand에는 standstill 금지
            "reward_vel_x": 1.5,
        })

    elif spec.mode == "slow_walk":
        terms += [
            "penalty_standstill = jnp.square(jnp.clip(target_vx - vb[0], 0.0, 10.0))",
            "penalty_lateral_vel = jnp.square(vb[1])",
            "penalty_yaw_rate = jnp.square(ab[2] - ang_vel_tar[2])",
            "penalty_all_feet_air = jnp.where(jnp.min(feet_z) > 0.03, 1.0, 0.0)",
            "penalty_feet_too_high = jnp.sum(jnp.square(jnp.clip(feet_z - 0.10, 0.0, 1.0)))",
        ]
        weights.update({
            "penalty_standstill": 1.0,
            "penalty_lateral_vel": 2.0,
            "penalty_yaw_rate": 1.5,
            "penalty_all_feet_air": 8.0,
            "penalty_feet_too_high": 3.0,
        })

    elif spec.mode == "walk":
        terms += [
            "penalty_standstill = jnp.square(jnp.clip(target_vx - vb[0], 0.0, 10.0))",
            "penalty_lateral_vel = jnp.square(vb[1])",
            "penalty_yaw_rate = jnp.square(ab[2] - ang_vel_tar[2])",
            "penalty_all_feet_air = jnp.where(jnp.min(feet_z) > 0.035, 1.0, 0.0)",
            "penalty_feet_too_high = jnp.sum(jnp.square(jnp.clip(feet_z - 0.12, 0.0, 1.0)))",
        ]
        weights.update({
            "reward_vel_x": 10.0,
            "penalty_standstill": 1.8,
            "penalty_lateral_vel": 2.5,
            "penalty_yaw_rate": 2.0,
            "penalty_all_feet_air": 6.0,
            "penalty_feet_too_high": 2.0,
        })

    else:  # trot_pace
        terms += [
            "penalty_standstill = jnp.square(jnp.clip(target_vx - vb[0], 0.0, 10.0))",
            "penalty_lateral_vel = jnp.square(vb[1])",
            "penalty_yaw_rate = jnp.square(ab[2] - ang_vel_tar[2])",
            "penalty_feet_too_high = jnp.sum(jnp.square(jnp.clip(feet_z - 0.16, 0.0, 1.0)))",
        ]
        weights.update({
            "reward_vel_x": 12.0,
            "penalty_standstill": 2.2,
            "penalty_lateral_vel": 2.0,
            "penalty_yaw_rate": 1.5,
            "penalty_feet_too_high": 1.0,
        })

    total_lines = []
    for name, w in weights.items():
        if name.startswith("reward_"):
            total_lines.append(f"        + {name} * {w}")
        else:
            total_lines.append(f"        - {name} * {w}")

    # 첫 줄 + 처리
    if total_lines:
        total_lines[0] = total_lines[0].replace("+ ", "", 1)

    reward_terms = "\n".join([f"    {x}" for x in terms])
    total_expr = "    total_reward = (\n" + "\n".join(total_lines) + "\n    )"

    code = f'''import jax
import jax.numpy as jnp
from brax import math
from dial_mpc.utils.function_utils import global_to_body_velocity


def compute_sds_reward(pipeline_state, state_info, env):
    """
    Auto-generated reward (core + motion head)
    mode={spec.mode}, gait={spec.gait}, target_speed={spec.target_speed_mps} m/s
    """
    torso_idx = env._torso_idx - 1

    torso_pos = pipeline_state.x.pos[torso_idx]
    torso_rot = pipeline_state.x.rot[torso_idx]
    torso_vel = pipeline_state.xd.vel[torso_idx]
    torso_ang = pipeline_state.xd.ang[torso_idx]

    feet_pos = pipeline_state.site_xpos[env._feet_site_id]
    feet_z = feet_pos[:, 2]

    joint_angles = pipeline_state.q[7:]
    joint_vel = pipeline_state.qvel[6:]
    ctrl = pipeline_state.ctrl
    default_pose = env._default_pose if hasattr(env, "_default_pose") else jnp.zeros_like(joint_angles)

    vel_tar = state_info["vel_tar"]
    ang_vel_tar = state_info["ang_vel_tar"]

    vb = global_to_body_velocity(torso_vel, torso_rot)
    # MJX에서는 보통 rad/s. deg->rad 변환하지 않음.
    ab = global_to_body_velocity(torso_ang, torso_rot)

    euler = math.quat_to_euler(torso_rot)
    roll, pitch, yaw = euler[0], euler[1], euler[2]

{reward_terms}

{total_expr}
    return total_reward
'''
    return code


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/final_sus_report.txt")
    parser.add_argument("--output", default="dial-mpc/dial_mpc/envs/sds_reward_function.py")
    parser.add_argument("--spec-out", default="output/motion_spec.json")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--no-api", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    in_path = (base / args.input).resolve() if not os.path.isabs(args.input) else Path(args.input)
    out_path = (base / args.output).resolve() if not os.path.isabs(args.output) else Path(args.output)
    spec_path = (base / args.spec_out).resolve() if not os.path.isabs(args.spec_out) else Path(args.spec_out)

    if not in_path.exists():
        raise FileNotFoundError(f"SUS report not found: {in_path}")

    report = read_text(in_path)

    spec = heuristic_spec(report)

    api_key = os.getenv("GEMINI_API_KEY", "")
    if (not args.no_api) and api_key:
        s = gemini_spec(report, model=args.model, api_key=api_key)
        if s is not None:
            spec = s

    code = build_reward_code(spec)
    write_text(out_path, code)
    write_text(spec_path, json.dumps(asdict(spec), ensure_ascii=False, indent=2))

    print(f"[OK] SUS input: {in_path}")
    print(f"[OK] Motion spec: {spec_path}")
    print(f"[OK] Reward code: {out_path}")
    print(f"[INFO] mode={spec.mode}, gait={spec.gait}, target_speed={spec.target_speed_mps} m/s")


if __name__ == "__main__":
    main()
