import jax
import jax.numpy as jnp
from brax import math
from dial_mpc.utils.function_utils import global_to_body_velocity


def compute_sds_reward(pipeline_state, state_info, env):
    """
    Auto-generated reward (core + motion head)
    mode=slow_walk, gait=walk, target_speed=0.45 m/s
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

    target_vx = jnp.where(jnp.abs(vel_tar[0]) < 1e-6, 0.450, vel_tar[0])
    reward_vel_x = -jnp.square(vb[0] - target_vx)
    penalty_orientation = jnp.square(roll) + jnp.square(pitch)
    penalty_height = jnp.square(torso_pos[2] - 0.300)
    penalty_vertical_vel = jnp.square(vb[2])
    penalty_accel = jnp.sum(jnp.square(ab))
    penalty_joint_pose = jnp.sum(jnp.square(joint_angles - default_pose))
    penalty_joint_vel = jnp.sum(jnp.square(joint_vel))
    penalty_energy = jnp.sum(jnp.square(ctrl))
    penalty_collapse = jnp.square(jnp.clip(0.26 - torso_pos[2], 0.0, 1.0))
    penalty_standstill = jnp.square(jnp.clip(target_vx - vb[0], 0.0, 10.0))
    penalty_lateral_vel = jnp.square(vb[1])
    penalty_yaw_rate = jnp.square(ab[2] - ang_vel_tar[2])
    penalty_all_feet_air = jnp.where(jnp.min(feet_z) > 0.03, 1.0, 0.0)
    penalty_feet_too_high = jnp.sum(jnp.square(jnp.clip(feet_z - 0.10, 0.0, 1.0)))

    total_reward = (
        reward_vel_x * 8.0
        - penalty_orientation * 8.0
        - penalty_height * 6.0
        - penalty_vertical_vel * 2.0
        - penalty_accel * 0.06
        - penalty_joint_pose * 0.12
        - penalty_joint_vel * 0.001
        - penalty_energy * 0.0003
        - penalty_collapse * 12.0
        - penalty_standstill * 1.0
        - penalty_lateral_vel * 2.0
        - penalty_yaw_rate * 1.5
        - penalty_all_feet_air * 8.0
        - penalty_feet_too_high * 3.0
    )
    return total_reward
