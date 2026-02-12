import os

# 파일 경로 설정
ENV_PATH = "dial-mpc/dial_mpc/envs/unitree_go2_env.py"
REWARD_CODE_PATH = "output/sds_reward_function.py"

def patch_environment():
    print(f"🔧 Reading reward code from {REWARD_CODE_PATH}...")
    if not os.path.exists(REWARD_CODE_PATH):
        print(f"❌ Error: Reward file not found!")
        return
        
    with open(REWARD_CODE_PATH, "r") as f:
        new_reward_code = f.read()

    print(f"🔧 Reading environment file from {ENV_PATH}...")
    if not os.path.exists(ENV_PATH):
        print(f"❌ Error: Environment file not found!")
        return

    with open(ENV_PATH, "r") as f:
        env_lines = f.readlines()

    # 1. 필수 Import 추가 (파일 맨 위에)
    # 이미 있는지 확인하고 없으면 추가
    imports_to_add = [
        "from dial_mpc.utils.function_utils import global_to_body_velocity",
        "import jax.numpy as jnp",
        "from brax import math",
        "import jax"
    ]
    
    final_lines = []
    # 기존 import 보존하면서 새 import 끼워넣기
    import_inserted = False
    for line in env_lines:
        final_lines.append(line)
        if (line.startswith("import") or line.startswith("from")) and not import_inserted:
            for imp in imports_to_add:
                if imp not in "".join(env_lines): # 파일 전체에 없으면 추가
                    final_lines.insert(0, imp + "\n")
            import_inserted = True

    # 2. 새로운 보상 함수(compute_sds_reward)를 파일 맨 끝에 추가
    final_lines.append("\n" + "#" * 40 + "\n")
    final_lines.append("# [INJECTED BY SDS] Generated Reward Function\n")
    final_lines.append("#" * 40 + "\n")
    final_lines.append(new_reward_code)
    final_lines.append("\n")

    # 3. 기존 get_reward 메서드를 찾아서, 새 함수를 호출하도록 변경
    # "def get_reward(self" 줄을 찾아서 그 다음 줄부터 return 문을 교체
    patched_lines = []
    in_get_reward = False
    
    for line in final_lines:
        if "def get_reward(self" in line:
            patched_lines.append(line)
            # 기존 get_reward 내부 로직을 무시하고 바로 새 함수 호출로 연결
            patched_lines.append("        # [Modified by SDS] Call injected reward function\n")
            patched_lines.append("        return compute_sds_reward(pipeline_state, state_info, self)\n")
            in_get_reward = True
        elif in_get_reward:
            # get_reward 함수가 끝날 때까지(들여쓰기가 없어질 때까지) 기존 코드 스킵
            # 빈 줄이나 주석은 무시하고, 들여쓰기가 8칸(공백)보다 적어지면 함수 끝난 걸로 간주
            if line.strip() and not line.startswith("        ") and not line.startswith("    #"):
                in_get_reward = False
                patched_lines.append(line) # 다음 함수나 클래스 시작
            else:
                pass # 기존 get_reward 내용 삭제 (스킵)
        else:
            patched_lines.append(line)

    # 4. 파일 덮어쓰기
    with open(ENV_PATH, "w") as f:
        f.writelines(patched_lines)
    
    print("✅ Success! 'unitree_go2_env.py' has been patched with the new reward function.")

if __name__ == "__main__":
    patch_environment()
