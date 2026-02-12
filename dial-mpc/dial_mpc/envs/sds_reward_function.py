import jax
import jax.numpy as jnp
from brax import math
from dial_mpc.utils.function_utils import global_to_body_velocity


def compute_sds_reward(pipeline_state, state_info, env):
    """
    Stability-first walk reward:
    - pitch/roll 강하게 억제
    - base height 유지
    - 관절 과접힘 억제 (home pose 기반)
    - 자세 안정 시에만 전진 보상 적용
    """
    torso_idx = env._torso_idx - 1

    torso_pos = pipeline_state.x.pos[torso_idx]
    torso_rot = pipeline_state.x.rot[torso_idx]
    torso_vel = pipeline_state.xd.vel[torso_idx]
    torso_ang = pipeline_state.xd.ang[torso_idx]

    joint_angles = pipeline_state.q[7:]
    joint_vel = pipeline_state.qvel[6:]
    ctrl = pipeline_state.ctrl

    vel_tar = state_info["vel_tar"]
    ang_vel_tar = state_info["ang_vel_tar"]

    vb = global_to_body_velocity(torso_vel, torso_rot)
    ab = global_to_body_velocity(torso_ang * jnp.pi / 180.0, torso_rot)

    euler = math.quat_to_euler(torso_rot)
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    # posture gate: 자세가 무너지면 전진 보상 거의 0
    posture_ok = jnp.exp(- (roll**2) / 0.5) * jnp.exp(- (pitch**2) / 0.5)

    # 기본 높이 유지
    penalty_height = jnp.square(torso_pos[2] - 0.30)

    # 자세 안정
    penalty_orientation = jnp.square(roll) + jnp.square(pitch)

    # 전진 속도 (자세 안정 시에만)
    reward_vel_x = -jnp.square(vb[0] - vel_tar[0]) * posture_ok
    penalty_vel_y = jnp.square(vb[1])
    reward_yaw_rate = -jnp.square(ab[2] - ang_vel_tar[2])

    # 수직 속도 억제
    penalty_vertical_vel = jnp.square(vb[2])
    # acceleration penalty (smooth start)
    penalty_accel = jnp.sum(jnp.square(torso_ang))

    # 관절 접힘 억제: home pose 기준 편차 패널티
    # env._default_pose exists in unitree_go2_env
    if hasattr(env, "_default_pose"):
        penalty_joint_pose = jnp.sum(jnp.square(joint_angles - env._default_pose))
    else:
        penalty_joint_pose = jnp.sum(jnp.square(joint_angles))

    penalty_joint_vel = jnp.sum(jnp.square(joint_vel))
    penalty_energy = jnp.sum(jnp.square(ctrl))

    # base 너무 낮아지는 것 강하게 억제
    penalty_collapse = jnp.square(jnp.clip(0.26 - torso_pos[2], 0.0, 1.0))

    total_reward = (
        reward_vel_x * 18.0
        - penalty_vel_y * 1.0
        + reward_yaw_rate * 0.5
        - penalty_height * 6.0
        - penalty_orientation * 8.0
        - penalty_vertical_vel * 1.5
        - penalty_accel * 0.05
        - penalty_joint_pose * 0.5
        - penalty_joint_vel * 0.002
        - penalty_standstill * 2.0
        - penalty_collapse * 10.0
        - penalty_energy * 0.0005
    )
    return total_reward
