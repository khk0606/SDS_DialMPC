import jax.numpy as jnp
from brax import math
from dial_mpc.utils.function_utils import global_to_body_velocity, get_foot_step


def compute_sds_reward(pipeline_state, state_info, env):
    """SDS reward with the original Dial-MPC Go2 scaffold preserved."""

    torso_idx = env._torso_idx - 1
    x = pipeline_state.x
    xd = pipeline_state.xd
    ctrl = pipeline_state.ctrl

    # gait/contact scaffold (same structure as original env reward)
    z_feet = pipeline_state.site_xpos[env._feet_site_id][:, 2]
    duty_ratio, cadence, amplitude = env._gait_params[env._gait]
    phases = env._gait_phase[env._gait]
    z_feet_tar = get_foot_step(
        duty_ratio, cadence, amplitude, phases, state_info["step"] * env.dt
    )
    reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)

    foot_pos = pipeline_state.site_xpos[env._feet_site_id]
    foot_contact_z = foot_pos[:, 2] - env._foot_radius
    contact = foot_contact_z < 1e-3
    contact_filt_mm = contact | state_info["last_contact"]
    first_contact = (state_info["feet_air_time"] > 0) * contact_filt_mm
    reward_air_time = jnp.sum((state_info["feet_air_time"] - 0.1) * first_contact)

    # targets
    vel_tar = state_info["vel_tar"]
    ang_vel_tar = state_info["ang_vel_tar"]
    target_vx = jnp.where(jnp.abs(vel_tar[0]) < 1e-6, 0.25, vel_tar[0])

    # kinematics
    pos = x.pos[torso_idx]
    rot = x.rot[torso_idx]
    vb = global_to_body_velocity(xd.vel[torso_idx], rot)
    ab = global_to_body_velocity(xd.ang[torso_idx], rot)

    # keep original reward blocks, adjust weights/targets for slow walk
    pos_tar = state_info["pos_tar"] + state_info["vel_tar"] * env.dt * state_info["step"]
    r_mat = math.quat_to_3x3(rot)
    head_vec = jnp.array([0.285, 0.0, 0.0])
    head_pos = pos + jnp.dot(r_mat, head_vec)
    reward_pos = -jnp.sum((head_pos - pos_tar) ** 2)

    vec_tar = jnp.array([0.0, 0.0, 1.0])
    vec = math.rotate(vec_tar, x.rot[0])
    reward_upright = -jnp.sum(jnp.square(vec - vec_tar))

    yaw_tar = state_info["yaw_tar"] + ang_vel_tar[2] * env.dt * state_info["step"]
    yaw = math.quat_to_euler(rot)[2]
    d_yaw = yaw - yaw_tar
    reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))

    reward_vel = -(jnp.square(vb[0] - target_vx) + 0.8 * jnp.square(vb[1]))
    reward_ang_vel = -jnp.square(ab[2] - ang_vel_tar[2])
    reward_height = -jnp.square(pos[2] - 0.30)

    power = jnp.maximum(ctrl * pipeline_state.qvel[6:] / 160.0, 0.0)
    penalty_energy = jnp.sum(power**2)
    penalty_joint_vel = jnp.sum(jnp.square(pipeline_state.qvel[6:]))
    penalty_vertical_vel = jnp.square(vb[2])
    penalty_collapse = jnp.square(jnp.clip(0.24 - pos[2], 0.0, 1.0))
    penalty_standstill = jnp.square(jnp.clip(target_vx - vb[0], 0.0, 10.0))
    penalty_all_feet_air = jnp.where(jnp.min(foot_contact_z) > 0.01, 1.0, 0.0)

    reward_alive = 1.0

    reward = (
        reward_gaits * 0.12
        + reward_air_time * 0.02
        + reward_pos * 0.05
        + reward_upright * 1.20
        + reward_yaw * 0.50
        + reward_vel * 2.00
        + reward_ang_vel * 0.60
        + reward_height * 1.20
        + reward_alive * 0.05
        - penalty_energy * 0.0005
        - penalty_joint_vel * 0.0008
        - penalty_vertical_vel * 0.60
        - penalty_collapse * 6.00
        - penalty_standstill * 0.80
        - penalty_all_feet_air * 1.50
    )
    return reward
