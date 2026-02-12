import requests
import json
import os
import re


API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")
# ==========================================
# API 설정
# ==========================================


# 파일 경로 및 모델 설정
MODEL_NAME = "gemini-2.5-flash"  # Gemini 2.5 Flash
INPUT_REPORT_PATH = os.path.join("output", "final_sus_report.txt")
OUTPUT_CODE_PATH = os.path.join("output", "sds_reward_function.py")

# 코드 템플릿
CODE_TEMPLATE = '''import jax
import jax.numpy as jnp
from brax import math
from dial_mpc.utils.function_utils import global_to_body_velocity


def compute_sds_reward(pipeline_state, state_info, env):
    """
    Compute SDS reward for quadruped locomotion.
    Auto-generated from Motion Analysis Report.
    """
    # ============================================
    # 1. Extract state variables
    # ============================================
    torso_idx = env._torso_idx - 1
    
    # Torso state
    torso_pos = pipeline_state.x.pos[torso_idx]
    torso_rot = pipeline_state.x.rot[torso_idx]
    torso_vel = pipeline_state.xd.vel[torso_idx]
    torso_ang = pipeline_state.xd.ang[torso_idx]
    
    # Feet positions
    feet_pos = pipeline_state.site_xpos[env._feet_site_id]
    feet_z = feet_pos[:, 2]
    
    # Joint state
    joint_angles = pipeline_state.q[7:]
    joint_vel = pipeline_state.qvel[6:]
    ctrl = pipeline_state.ctrl
    
    # Targets from state_info
    vel_tar = state_info["vel_tar"]
    ang_vel_tar = state_info["ang_vel_tar"]
    pos_tar = state_info["pos_tar"]
    
    # Body-frame velocity
    vb = global_to_body_velocity(torso_vel, torso_rot)
    ab = global_to_body_velocity(torso_ang * jnp.pi / 180.0, torso_rot)
    
    # Orientation (euler angles)
    euler = math.quat_to_euler(torso_rot)
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    
    # ============================================
    # 2. Reward terms
    # ============================================
{reward_terms}
    
    # ============================================
    # 3. Total reward
    # ============================================
{total_reward}
    
    return total_reward
'''

# 시스템 프롬프트
SYSTEM_PROMPT = """You are a robotics reward function engineer for quadruped robots.

TASK: Generate reward terms for a JAX-based reward function.

AVAILABLE VARIABLES (already defined, do NOT redefine):
- vb: [3] body-frame velocity (vb[0]=forward, vb[1]=lateral, vb[2]=vertical)
- ab: [3] body-frame angular velocity
- torso_pos: [3] base position (torso_pos[2]=height)
- roll, pitch, yaw: euler angles (scalars)
- feet_z: [4] feet z-positions
- joint_angles: [12] joint positions
- joint_vel: [12] joint velocities  
- ctrl: [12] control signals
- vel_tar: [3] target velocity
- ang_vel_tar: [3] target angular velocity
- pos_tar: [3] target position

OUTPUT FORMAT - Return exactly TWO blocks separated by ---SPLIT---:

BLOCK 1 (reward_terms):
    # Velocity tracking (forward)
    reward_vel_x = -jnp.square(vb[0] - 10.0)
    
    # Lateral velocity penalty
    penalty_vel_y = jnp.square(vb[1])
    
    # Height maintenance
    reward_height = -jnp.square(torso_pos[2] - 0.3)
    
    # Upright reward
    reward_upright = -jnp.square(roll) - jnp.square(pitch) * 0.5
    
    # Energy penalty
    penalty_energy = jnp.sum(jnp.square(ctrl))
    
    # Aerial phase bonus (for gallop)
    min_feet_z = jnp.min(feet_z)
    reward_aerial = jnp.where(min_feet_z > 0.02, 1.0, 0.0)

---SPLIT---

BLOCK 2 (total_reward):
    total_reward = (
        reward_vel_x * 1.0
        + reward_height * 0.5
        + reward_upright * 0.3
        + reward_aerial * 0.2
        - penalty_vel_y * 0.5
        - penalty_energy * 0.001
    )

RULES:
1. Use ONLY jnp functions (jnp.square, jnp.sum, jnp.where, jnp.exp, jnp.clip, jnp.min, jnp.max)
2. All terms must be SCALARS
3. Indent with 4 spaces
4. Keep it simple - 5 to 8 reward terms is enough
5. MUST include ---SPLIT--- between the two blocks
6. Do NOT add any text or comments after ---SPLIT--- line, just the code block"""


def read_sus_report():
    paths_to_check = [
        INPUT_REPORT_PATH,
        "final_sus_report.txt",
        "sus_report.txt",
    ]
    for path in paths_to_check:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return None


def validate_code_block(code):
    """괄호 매칭 검사"""
    parens = {'(': ')', '[': ']', '{': '}'}
    stack = []
    for char in code:
        if char in parens:
            stack.append(char)
        elif char in parens.values():
            if not stack:
                return False, "Unmatched closing bracket"
            if parens[stack.pop()] != char:
                return False, "Mismatched brackets"
    if stack:
        return False, f"Unclosed brackets: {stack}"
    return True, "OK"


def fix_indent(code, base_indent=4):
    """들여쓰기 정리"""
    lines = code.strip().split('\n')
    fixed_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped and not stripped.startswith('#'):
            fixed_lines.append(' ' * base_indent + stripped)
        elif stripped.startswith('#'):
            fixed_lines.append(' ' * base_indent + stripped)
        else:
            fixed_lines.append('')
    return '\n'.join(fixed_lines)


def generate_code():
    print(">>> Reading Motion Analysis Report...")
    sus_report = read_sus_report()

    if not sus_report:
        print("[ERROR] Could not find report file.")
        return False

    print(f"[OK] Report loaded ({len(sus_report)} chars).")
    print(f">>> Requesting reward terms from Gemini ({MODEL_NAME})...")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}

    user_prompt = f"""
Motion Analysis Report:
{sus_report}

Generate reward_terms and total_reward for this motion.
Key requirements from report:
- Target velocity range
- Gait characteristics (aerial phase, pitch oscillation, etc.)
- Stability constraints

Output the two code blocks separated by ---SPLIT---"""

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
            {"role": "model", "parts": [{"text": "I will generate the reward terms following the exact format with ---SPLIT--- separator."}]},
            {"role": "user", "parts": [{"text": user_prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192
        }
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code != 200:
            print(f"[ERROR] API Error {response.status_code}: {response.text}")
            return False

        result = response.json()
        try:
            content_text = result['candidates'][0]['content']['parts'][0]['text']
        except KeyError:
            print("[ERROR] Unexpected response format.")
            print(result)
            return False

        # 마크다운 제거
        content_text = re.sub(r'```python\s*', '', content_text)
        content_text = re.sub(r'```\s*', '', content_text)

        # SPLIT 파싱
        if "---SPLIT---" not in content_text:
            print("[ERROR] Gemini did not include ---SPLIT--- separator.")
            print("[DEBUG] Raw output:")
            print(content_text[:500])
            return False

        parts = content_text.split("---SPLIT---")
        reward_terms = parts[0].strip()
        total_reward = parts[1].strip() if len(parts) > 1 else ""

        # 검증
        valid1, msg1 = validate_code_block(reward_terms)
        valid2, msg2 = validate_code_block(total_reward)

        if not valid1:
            print(f"[ERROR] reward_terms validation failed: {msg1}")
            print("[DEBUG] reward_terms:")
            print(reward_terms)
            return False

        if not valid2:
            print(f"[ERROR] total_reward validation failed: {msg2}")
            print("[DEBUG] total_reward:")
            print(total_reward)
            return False

        # total_reward에 실제 계산이 있는지 확인
        if "total_reward = 0" in total_reward or "total_reward=0" in total_reward:
            print("[ERROR] total_reward is just 0.0, not using reward terms.")
            return False

        # 들여쓰기 정리
        reward_terms = fix_indent(reward_terms, 4)
        total_reward = fix_indent(total_reward, 4)

        # 템플릿에 삽입
        final_code = CODE_TEMPLATE.format(
            reward_terms=reward_terms,
            total_reward=total_reward
        )

        # 최종 문법 검사
        try:
            compile(final_code, '<string>', 'exec')
        except SyntaxError as e:
            print(f"[ERROR] Syntax error in generated code: {e}")
            print("[DEBUG] Generated code:")
            print(final_code)
            return False

        # 저장
        output_dir = os.path.dirname(OUTPUT_CODE_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(OUTPUT_CODE_PATH, "w", encoding="utf-8") as f:
            f.write(final_code)

        print("\n" + "=" * 60)
        print(f"[SUCCESS] Reward function saved to: {OUTPUT_CODE_PATH}")
        print("=" * 60)

        # 검증 결과
        print("\n[Validation Checks]")
        checks = [
            ("Function signature", "def compute_sds_reward(pipeline_state, state_info, env)" in final_code),
            ("env._torso_idx", "env._torso_idx" in final_code),
            ("env._feet_site_id", "env._feet_site_id" in final_code),
            ("jnp usage", "jnp." in final_code),
            ("return total_reward", "return total_reward" in final_code),
            ("Syntax OK", True),
        ]

        for name, passed in checks:
            print(f"  [{'OK' if passed else 'FAIL'}] {name}")

        print("\n[Generated Code]")
        print("-" * 60)
        print(final_code)
        print("-" * 60)
        
        return True

    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = generate_code()
    if not success:
        print("\n[RETRY] Generation failed. Please run again.")