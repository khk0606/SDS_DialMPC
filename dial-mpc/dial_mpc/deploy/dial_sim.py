import json
import os
import time
from dataclasses import dataclass
import importlib
from multiprocessing import shared_memory
from pathlib import Path
import sys

import argparse
import art
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import scienceplots
import yaml

from dial_mpc.config.base_env_config import BaseEnvConfig
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.examples import deploy_examples
from dial_mpc.utils.io_utils import (
    get_example_path,
    get_model_path,
    load_dataclass_from_dict,
)

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
    headless: bool = False
    max_time_sec: float = 0.0
    max_steps: int = 0
    stop_on_fall: bool = False
    fall_height_threshold: float = 0.18
    target_base_height: float = 0.30
    metrics_filename: str = "metrics.json"


class DialSim:
    def __init__(
        self,
        sim_config: DialSimConfig,
        env_config: BaseEnvConfig,
        dial_config: DialConfig,
        config_dict: dict | None = None,
    ):
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
        self.headless = bool(sim_config.headless)

        self.max_time_sec = float(sim_config.max_time_sec)
        self.max_steps = int(sim_config.max_steps)
        self.stop_on_fall = bool(sim_config.stop_on_fall)
        self.fall_height_threshold = float(sim_config.fall_height_threshold)
        self.target_base_height = float(sim_config.target_base_height)
        self.metrics_filename = str(sim_config.metrics_filename)

        self.step_count = 0
        self.stop_reason = "running"
        self.sim_exception: str | None = None
        self.config_dict = dict(config_dict or {})

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
        self.home_roll = 0.0
        self.home_pitch = 0.0
        if self.default_q.shape[0] >= 7:
            home_quat = np.asarray(self.default_q[3:7], dtype=np.float64).reshape(1, 4)
            hr, hp = self._roll_pitch_from_quat(home_quat)
            self.home_roll = float(hr[0])
            self.home_pitch = float(hp[0])

        # position cmd smoothing (prevents start spike)
        self.enable_cmd_slew = True
        self.cmd_slew_step = 0.02
        self.prev_pos_cmd = self.default_q[7 : 7 + self.Nu].copy()

        self.ctrl_low = None
        self.ctrl_high = None
        if self.mj_model.actuator_ctrlrange.shape[0] == self.Nu:
            ctrl_low = self.mj_model.actuator_ctrlrange[:, 0].copy()
            ctrl_high = self.mj_model.actuator_ctrlrange[:, 1].copy()
            ctrl_span = np.abs(ctrl_high - ctrl_low)
            # Some models expose [0, 0] ctrlrange for all actuators when limits are not used.
            # In that case clipping would collapse all commands to zero.
            if np.any(ctrl_span > 1e-9):
                self.ctrl_low = ctrl_low
                self.ctrl_high = ctrl_high

        # publisher
        self.time_shm = shared_memory.SharedMemory(name="time_shm", create=True, size=32)
        self.time_shared = np.ndarray(1, dtype=np.float32, buffer=self.time_shm.buf)
        self.time_shared[0] = 0.0

        self.state_shm = shared_memory.SharedMemory(name="state_shm", create=True, size=self.Nx * 32)
        self.state_shared = np.ndarray((self.Nx,), dtype=np.float32, buffer=self.state_shm.buf)

        # listener
        self.acts_shm = shared_memory.SharedMemory(name="acts_shm", create=True, size=self.n_acts * self.Nu * 32)
        self.acts_shared = np.ndarray((self.n_acts, self.mj_model.nu), dtype=np.float32, buffer=self.acts_shm.buf)
        self.acts_shared[:] = self.default_q[7 : 7 + self.Nu]

        self.refs_shm = shared_memory.SharedMemory(
            name="refs_shm", create=True, size=self.n_acts * self.Nu * 3 * 32
        )
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

    def _append_record(self):
        self.data.append(np.concatenate([[self.t], self.mj_data.qpos, self.mj_data.qvel, self.mj_data.ctrl]))

    def _should_stop(self) -> tuple[bool, str]:
        if self.max_steps > 0 and self.step_count >= self.max_steps:
            return True, "max_steps"
        if self.max_time_sec > 0.0 and self.t >= self.max_time_sec:
            return True, "max_time_sec"
        if self.stop_on_fall and float(self.mj_data.qpos[2]) < self.fall_height_threshold:
            return True, "fall_detected"
        return False, ""

    def _extract_targets(self) -> tuple[float, float, float]:
        target_vx = 0.0
        target_wz = 0.0
        target_height = self.target_base_height

        fixed_vel = self.config_dict.get("fixed_vel_tar")
        if isinstance(fixed_vel, list) and len(fixed_vel) >= 1:
            target_vx = float(fixed_vel[0])
        elif "default_vx" in self.config_dict:
            target_vx = float(self.config_dict.get("default_vx", 0.0))

        fixed_ang = self.config_dict.get("fixed_ang_vel_tar")
        if isinstance(fixed_ang, list) and len(fixed_ang) >= 3:
            target_wz = float(fixed_ang[2])
        elif "default_vyaw" in self.config_dict:
            target_wz = float(self.config_dict.get("default_vyaw", 0.0))

        reward_cfg = self.config_dict.get("reward")
        if isinstance(reward_cfg, dict) and "target_base_height" in reward_cfg:
            target_height = float(reward_cfg.get("target_base_height", target_height))
        elif "target_base_height" in self.config_dict:
            target_height = float(self.config_dict.get("target_base_height", target_height))

        return target_vx, target_wz, target_height

    @staticmethod
    def _roll_pitch_from_quat(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # MuJoCo qpos quaternion order: [w, x, y, z]
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        return roll, pitch

    @staticmethod
    def _angle_dev(angle: np.ndarray, ref: float) -> np.ndarray:
        diff = angle - ref
        return np.arctan2(np.sin(diff), np.cos(diff))

    def compute_metrics(self) -> dict:
        if len(self.data) == 0:
            return {
                "success": False,
                "fail_reason": self.stop_reason if self.stop_reason != "running" else "no_data",
                "fall_time_sec": -1.0,
                "fall_count": 0,
                "mean_roll": 0.0,
                "mean_pitch": 0.0,
                "mean_height_error": 0.0,
                "mean_vel_error": 0.0,
                "yaw_rate_error": 0.0,
                "contact_match_score": 1.0,
                "energy": 0.0,
                "episode_return": -1e6,
                "steps": int(self.step_count),
                "sim_time_sec": float(self.t),
                "stop_reason": self.stop_reason,
            }

        data = np.array(self.data, dtype=np.float64)
        nq = self.mj_model.nq
        nv = self.mj_model.nv

        t_series = data[:, 0]
        qpos = data[:, 1 : 1 + nq]
        qvel = data[:, 1 + nq : 1 + nq + nv]
        ctrl = data[:, 1 + nq + nv :]

        base_z = qpos[:, 2] if nq > 2 else np.zeros(data.shape[0], dtype=np.float64)
        fall_mask = base_z < self.fall_height_threshold
        prev_mask = np.concatenate(([False], fall_mask[:-1]))
        fall_events = np.logical_and(fall_mask, np.logical_not(prev_mask))
        fall_count = int(fall_events.sum())
        fall_time_sec = float(t_series[np.argmax(fall_events)]) if fall_count > 0 else -1.0

        if nq >= 7:
            quat = qpos[:, 3:7]
            roll, pitch = self._roll_pitch_from_quat(quat)
            mean_roll = float(np.mean(np.abs(roll)))
            mean_pitch = float(np.mean(np.abs(pitch)))
            roll_dev = self._angle_dev(roll, self.home_roll)
            pitch_dev = self._angle_dev(pitch, self.home_pitch)
            mean_roll_dev = float(np.mean(np.abs(roll_dev)))
            mean_pitch_dev = float(np.mean(np.abs(pitch_dev)))
            max_pitch_dev = float(np.max(np.abs(pitch_dev)))
        else:
            mean_roll = 0.0
            mean_pitch = 0.0
            mean_roll_dev = 0.0
            mean_pitch_dev = 0.0
            max_pitch_dev = 0.0

        target_vx, target_wz, target_height = self._extract_targets()

        actual_vx = qvel[:, 0] if nv > 0 else np.zeros(data.shape[0], dtype=np.float64)
        actual_wz = qvel[:, 5] if nv > 5 else np.zeros(data.shape[0], dtype=np.float64)
        mean_vel_error = float(np.mean(np.abs(actual_vx - target_vx)))
        yaw_rate_error = float(np.mean(np.abs(actual_wz - target_wz)))
        mean_height_error = float(np.mean(np.abs(base_z - target_height)))

        if nv > 6:
            joint_vel = qvel[:, 6 : 6 + self.Nu]
        else:
            joint_vel = np.zeros((data.shape[0], 0), dtype=np.float64)

        if joint_vel.shape[1] > 0 and ctrl.shape[1] > 0:
            m = min(joint_vel.shape[1], ctrl.shape[1])
            power = ctrl[:, :m] * joint_vel[:, :m]
            energy = float(np.mean(np.sum(np.square(power), axis=1)))
        else:
            energy = 0.0

        success = bool((fall_count == 0) and (self.sim_exception is None))
        fail_reason = ""
        if not success:
            if self.sim_exception is not None:
                fail_reason = "sim_exception"
            elif fall_count > 0:
                fail_reason = "fall_detected"
            else:
                fail_reason = self.stop_reason if self.stop_reason != "running" else "unknown"

        episode_return = float(
            -(
                5.0 * mean_vel_error
                + 2.0 * mean_height_error
                + 0.5 * (mean_roll + mean_pitch)
                + 0.01 * energy
            )
        )

        return {
            "success": success,
            "fail_reason": fail_reason,
            "fall_time_sec": fall_time_sec,
            "fall_count": fall_count,
            "mean_roll": mean_roll,
            "mean_pitch": mean_pitch,
            "mean_roll_dev": mean_roll_dev,
            "mean_pitch_dev": mean_pitch_dev,
            "max_pitch_dev": max_pitch_dev,
            "mean_height_error": mean_height_error,
            "mean_vel_error": mean_vel_error,
            "yaw_rate_error": yaw_rate_error,
            "contact_match_score": 1.0,
            "energy": energy,
            "episode_return": episode_return,
            "steps": int(self.step_count),
            "sim_time_sec": float(self.t),
            "stop_reason": self.stop_reason,
        }

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

        viewer = None
        if not self.headless:
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

            if self.draw_refs and viewer is not None:
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
                q = self.mj_data.qpos
                while self.t <= (self.plan_time_shared[0] + self.ctrl_dt):
                    if self.leg_control == "position":
                        self.mj_data.ctrl = self._safe_position_cmd(self.acts_shared[0])
                    elif self.leg_control == "torque":
                        self.mj_data.ctrl = self.tau_shared[0]

                    if self.record:
                        self._append_record()

                    mujoco.mj_step(self.mj_model, self.mj_data)
                    self.t += self.sim_dt
                    self.step_count += 1

                    q = self.mj_data.qpos
                    qd = self.mj_data.qvel
                    state = np.concatenate([q, qd])
                    self.time_shared[:] = self.t
                    self.state_shared[:] = state

                    should_stop, reason = self._should_stop()
                    if should_stop:
                        self.stop_reason = reason
                        break

                self.q_history = np.roll(self.q_history, -1, axis=0)
                self.q_history[-1, :] = q[7:]
                self.qref_history = np.roll(self.qref_history, -1, axis=0)
                self.qref_history[-1, :] = self.mj_data.ctrl
                if viewer is not None:
                    viewer.sync()

                if self.stop_reason != "running":
                    break

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
                    self._append_record()

                mujoco.mj_step(self.mj_model, self.mj_data)
                self.t += self.sim_dt
                self.step_count += 1

                q = self.mj_data.qpos
                qd = self.mj_data.qvel
                state = np.concatenate([q, qd])

                self.time_shared[:] = self.t
                self.state_shared[:] = state

                self.q_history = np.roll(self.q_history, -1, axis=0)
                self.q_history[-1, :] = q[7:]
                self.qref_history = np.roll(self.qref_history, -1, axis=0)
                self.qref_history[-1, :] = self.mj_data.ctrl
                if viewer is not None:
                    viewer.sync()

                should_stop, reason = self._should_stop()
                if should_stop:
                    self.stop_reason = reason
                    break

                duration = time.time() - t0
                if duration < self.sim_dt / self.real_time_factor:
                    time.sleep(self.sim_dt / self.real_time_factor - duration)
                else:
                    print("[WARN] Sim loop overruns")

        if self.stop_reason == "running":
            self.stop_reason = "completed"

    @staticmethod
    def _safe_close_unlink(shm_obj):
        try:
            shm_obj.close()
        except Exception:
            pass
        try:
            shm_obj.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def close(self):
        self._safe_close_unlink(self.time_shm)
        self._safe_close_unlink(self.state_shm)
        self._safe_close_unlink(self.acts_shm)
        self._safe_close_unlink(self.plan_time_shm)
        self._safe_close_unlink(self.refs_shm)
        self._safe_close_unlink(self.tau_shm)


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
        config_path = get_example_path(args.example + ".yaml")
    else:
        config_path = os.path.abspath(args.config)

    print(f"[CONFIG] dial_sim.py using: {config_path}")
    config_dict = yaml.safe_load(open(config_path, "r"))

    sim_config = load_dataclass_from_dict(DialSimConfig, config_dict)
    env_config = load_dataclass_from_dict(BaseEnvConfig, config_dict)
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    mujoco_env = DialSim(sim_config, env_config, dial_config, config_dict=config_dict)

    try:
        mujoco_env.main_loop()
    except KeyboardInterrupt:
        mujoco_env.stop_reason = "keyboard_interrupt"
    except Exception as exc:
        mujoco_env.sim_exception = str(exc)
        mujoco_env.stop_reason = "sim_exception"
        raise
    finally:
        base_output_dir = Path(dial_config.output_dir)
        base_output_dir.mkdir(parents=True, exist_ok=True)

        artifact_dir = None
        if mujoco_env.record and len(mujoco_env.data) > 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            data = np.array(mujoco_env.data)
            artifact_dir = base_output_dir / f"sim_{dial_config.env_name}_{env_config.task_name}_{timestamp}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            np.save(artifact_dir / "states.npy", data)

        metrics = mujoco_env.compute_metrics()
        if artifact_dir is not None:
            metrics["artifact_dir"] = str(artifact_dir)

        metrics_path = base_output_dir / sim_config.metrics_filename
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[METRICS] wrote {metrics_path}")

        mujoco_env.close()


if __name__ == "__main__":
    main()
