import os
import time
from multiprocessing import shared_memory
from dataclasses import dataclass
import importlib
import sys

import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import art

import mujoco
import mujoco.viewer

from dial_mpc.config.base_env_config import BaseEnvConfig
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.utils.io_utils import (
    load_dataclass_from_dict,
    get_model_path,
    get_example_path,
)
from dial_mpc.examples import deploy_examples

plt.style.use(["science"])


@dataclass
class DialSimConfig:
    robot_name: str
    scene_name: str
    sim_leg_control: str
    plot: bool
    record: bool
    real_time_factor: float
    sim_dt: float
    sync_mode: bool
    draw_refs: bool = False


class DialSim:
    def __init__(self, sim_config: DialSimConfig, env_config: BaseEnvConfig, dial_config: DialConfig):
        self.plot = sim_config.plot
        self.record = sim_config.record
        self.data = []
        self.ctrl_dt = env_config.dt
        self.real_time_factor = sim_config.real_time_factor
        self.sim_dt = sim_config.sim_dt
        self.n_acts = dial_config.Hsample + 1
        self.n_frame = int(self.ctrl_dt / self.sim_dt)
        self.t = 0.0
        self.sync_mode = sim_config.sync_mode
        self.leg_control = sim_config.sim_leg_control
        self.draw_refs = bool(sim_config.draw_refs)

        self.mj_model = mujoco.MjModel.from_xml_path(
            get_model_path(sim_config.robot_name, sim_config.scene_name).as_posix()
        )
        self.mj_model.opt.timestep = self.sim_dt
        self.mj_data = mujoco.MjData(self.mj_model)

        self.q_history = np.zeros((self.n_acts, self.mj_model.nu))
        self.qref_history = np.zeros((self.n_acts, self.mj_model.nu))
        self.n_plot_joint = 4

        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.Nx = self.mj_model.nq + self.mj_model.nv
        self.Nu = self.mj_model.nu
        self.default_q = self.mj_model.keyframe("home").qpos
        self.default_u = self.mj_model.keyframe("home").ctrl

        # position cmd smoothing (prevents start spike)
        self.enable_cmd_slew = True
        self.cmd_slew_step = 0.02
        self.prev_pos_cmd = self.default_q[7:7 + self.Nu].copy()

        if self.mj_model.actuator_ctrlrange.shape[0] == self.Nu:
            self.ctrl_low = self.mj_model.actuator_ctrlrange[:, 0].copy()
            self.ctrl_high = self.mj_model.actuator_ctrlrange[:, 1].copy()
        else:
            self.ctrl_low = None
            self.ctrl_high = None

        # publisher
        self.time_shm = shared_memory.SharedMemory(name="time_shm", create=True, size=32)
        self.time_shared = np.ndarray(1, dtype=np.float32, buffer=self.time_shm.buf)
        self.time_shared[0] = 0.0

        self.state_shm = shared_memory.SharedMemory(name="state_shm", create=True, size=self.Nx * 32)
        self.state_shared = np.ndarray((self.Nx,), dtype=np.float32, buffer=self.state_shm.buf)

        # listener
        self.acts_shm = shared_memory.SharedMemory(name="acts_shm", create=True, size=self.n_acts * self.Nu * 32)
        self.acts_shared = np.ndarray((self.n_acts, self.mj_model.nu), dtype=np.float32, buffer=self.acts_shm.buf)
        self.acts_shared[:] = self.default_q[7:7 + self.Nu]

        self.refs_shm = shared_memory.SharedMemory(name="refs_shm", create=True, size=self.n_acts * self.Nu * 3 * 32)
        self.refs_shared = np.ndarray((self.n_acts, self.Nu, 3), dtype=np.float32, buffer=self.refs_shm.buf)
        self.refs_shared[:] = 0.0

        self.plan_time_shm = shared_memory.SharedMemory(name="plan_time_shm", create=True, size=32)
        self.plan_time_shared = np.ndarray(1, dtype=np.float32, buffer=self.plan_time_shm.buf)
        self.plan_time_shared[0] = -self.ctrl_dt

        self.tau_shm = shared_memory.SharedMemory(name="tau_shm", create=True, size=self.n_acts * self.Nu * 32)
        self.tau_shared = np.ndarray((self.n_acts, self.mj_model.nu), dtype=np.float32, buffer=self.tau_shm.buf)
        self.tau_shared[:] = 0.0

    def _safe_position_cmd(self, target):
        cmd = np.asarray(target, dtype=np.float32).copy()

        if self.ctrl_low is not None and self.ctrl_high is not None:
            cmd = np.clip(cmd, self.ctrl_low, self.ctrl_high)

        if self.enable_cmd_slew:
            delta = cmd - self.prev_pos_cmd
            delta = np.clip(delta, -self.cmd_slew_step, self.cmd_slew_step)
            cmd = self.prev_pos_cmd + delta
            self.prev_pos_cmd = cmd.copy()

        return cmd

    def main_loop(self):
        if self.plot:
            fig, axs = plt.subplots(self.n_plot_joint, 1, figsize=(12, 12))
            handles = []
            handles_ref = []
            colors = plt.cm.rainbow(np.linspace(0, 1, self.n_plot_joint))
            for i in range(self.n_plot_joint):
                handles.append(axs[i].plot(self.q_history[:, i], color=colors[i])[0])
                handles_ref.append(axs[i].plot(self.qref_history[:, i], color=colors[i], linestyle="--")[0])
                axs[i].set_ylim(-1.0 + self.default_q[i + 7], 1.0 + self.default_q[i + 7])
                axs[i].set_xlabel("Time (s)")
                axs[i].set_ylabel(f"Joint {i+1} Position")
            plt.show(block=False)

        viewer = mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
        )

        if self.draw_refs:
            cnt = 0
            viewer.user_scn.ngeom = 0
            for i in range(self.n_acts - 1):
                for j in range(self.mj_model.nu):
                    color = np.array(
                        [1.0 * i / max(1, (self.n_acts - 1)), 1.0 * j / max(1, self.mj_model.nu), 0.0, 1.0]
                    )
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[cnt],
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=np.zeros(3),
                        rgba=color,
                        pos=self.refs_shared[i, j, :],
                        mat=np.eye(3).flatten(),
                    )
                    cnt += 1
            viewer.user_scn.ngeom = cnt
        else:
            viewer.user_scn.ngeom = 0
        viewer.sync()

        while True:
            if self.plot:
                for j in range(self.n_plot_joint):
                    handles[j].set_ydata(self.acts_shared[:, j])
                    handles_ref[j].set_ydata(self.qref_history[:, j])
                plt.pause(0.001)

            if self.draw_refs:
                for i in range(self.n_acts - 1):
                    for j in range(self.mj_model.nu):
                        r0 = self.refs_shared[i, j, :]
                        r1 = self.refs_shared[i + 1, j, :]
                        mujoco.mjv_connector(
                            viewer.user_scn.geoms[i * self.mj_model.nu + j],
                            mujoco.mjtGeom.mjGEOM_CAPSULE,
                            0.02,
                            r0,
                            r1,
                        )

            if self.sync_mode:
                while self.t <= (self.plan_time_shared[0] + self.ctrl_dt):
                    if self.leg_control == "position":
                        self.mj_data.ctrl = self._safe_position_cmd(self.acts_shared[0])
                    elif self.leg_control == "torque":
                        self.mj_data.ctrl = self.tau_shared[0]

                    if self.record:
                        self.data.append(
                            np.concatenate([[self.t], self.mj_data.qpos, self.mj_data.qvel, self.mj_data.ctrl])
                        )

                    mujoco.mj_step(self.mj_model, self.mj_data)
                    self.t += self.sim_dt

                    q = self.mj_data.qpos
                    qd = self.mj_data.qvel
                    state = np.concatenate([q, qd])
                    self.time_shared[:] = self.t
                    self.state_shared[:] = state

                self.q_history = np.roll(self.q_history, -1, axis=0)
                self.q_history[-1, :] = q[7:]
                self.qref_history = np.roll(self.qref_history, -1, axis=0)
                self.qref_history[-1, :] = self.mj_data.ctrl
                viewer.sync()

            else:
                t0 = time.time()
                if self.plan_time_shared[0] < 0.0:
                    time.sleep(0.01)
                    continue

                delta_time = self.t - self.plan_time_shared[0]
                delta_step = int(delta_time / self.ctrl_dt)
                if delta_time > self.ctrl_dt / self.real_time_factor:
                    print(f"[WARN] Delayed by {delta_time * 1000.0:.1f} ms")
                if delta_step >= self.n_acts or delta_step < 0:
                    delta_step = self.n_acts - 1

                if self.leg_control == "position":
                    self.mj_data.ctrl = self._safe_position_cmd(self.acts_shared[delta_step])
                elif self.leg_control == "torque":
                    self.mj_data.ctrl = self.tau_shared[delta_step]

                if self.record:
                    self.data.append(
                        np.concatenate([[self.t], self.mj_data.qpos, self.mj_data.qvel, self.mj_data.ctrl])
                    )

                mujoco.mj_step(self.mj_model, self.mj_data)
                self.t += self.sim_dt

                q = self.mj_data.qpos
                qd = self.mj_data.qvel
                state = np.concatenate([q, qd])

                self.time_shared[:] = self.t
                self.state_shared[:] = state

                self.q_history = np.roll(self.q_history, -1, axis=0)
                self.q_history[-1, :] = q[7:]
                self.qref_history = np.roll(self.qref_history, -1, axis=0)
                self.qref_history[-1, :] = self.mj_data.ctrl
                viewer.sync()

                duration = time.time() - t0
                if duration < self.sim_dt / self.real_time_factor:
                    time.sleep(self.sim_dt / self.real_time_factor - duration)
                else:
                    print("[WARN] Sim loop overruns")

    def close(self):
        self.time_shm.close()
        self.time_shm.unlink()
        self.state_shm.close()
        self.state_shm.unlink()
        self.acts_shm.close()
        self.acts_shm.unlink()
        self.plan_time_shm.close()
        self.plan_time_shm.unlink()
        self.refs_shm.close()
        self.refs_shm.unlink()
        self.tau_shm.close()
        self.tau_shm.unlink()


def main(args=None):
    art.tprint("LeCAR @ CMU\nDIAL-MPC\nSIMULATOR", font="big", chr_ignore=True)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    group.add_argument("--example", type=str, default=None, help="Example to run")
    group.add_argument("--list-examples", action="store_true", help="List available examples")
    parser.add_argument("--custom-env", type=str, default=None, help="Custom environment to import dynamically")
    args = parser.parse_args(args)

    if args.custom_env is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)

    if args.list_examples:
        print("Available examples:")
        for example in deploy_examples:
            print(f"  - {example}")
        return

    if args.example is not None:
        if args.example not in deploy_examples:
            print(f"Example {args.example} not found.")
            return
        config_dict = yaml.safe_load(open(get_example_path(args.example + ".yaml"), "r"))
    else:
        config_dict = yaml.safe_load(open(args.config, "r"))

    sim_config = load_dataclass_from_dict(DialSimConfig, config_dict)
    env_config = load_dataclass_from_dict(BaseEnvConfig, config_dict)
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    mujoco_env = DialSim(sim_config, env_config, dial_config)

    try:
        mujoco_env.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        if mujoco_env.record:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            data = np.array(mujoco_env.data)
            output_dir = os.path.join(
                dial_config.output_dir,
                f"sim_{dial_config.env_name}_{env_config.task_name}_{timestamp}",
            )
            os.makedirs(output_dir)
            np.save(os.path.join(output_dir, "states"), data)

        mujoco_env.close()


if __name__ == "__main__":
    main()
