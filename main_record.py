from __future__ import annotations

import time
from pathlib import Path

import example_robot_data as erd
import numpy as np

from mpc import Gait, MPC, MPCMeshcatVisualizer, QuadKinodynOcp, QuadRobot
from record.config import RecordConfig
from record.trajectory import TrajectoryGenerator
from record.yaml_recording import YamlRecorder


def build_robot(record_config: RecordConfig):
    viewer_robot = erd.load(record_config.robot_name)
    robot = QuadRobot(
        viewer_robot.model,
        record_config.reference_configuration_name,
        record_config.base_joint_name,
    )
    for foot_name in record_config.foot_names:
        robot.addPointFoot(foot_name, record_config.base_joint_name)
    return viewer_robot, robot


def build_kinodynamics_problem(robot, record_config: RecordConfig):
    gravity = np.array([0.0, 0.0, -9.81])
    nv = robot.getModel().nv
    w_basepos = [10, 10, 200, 50, 50, 10]
    w_legpos = [0.2, 0.2, 0.1]
    w_basevel = [10, 10, 100, 10, 10, 10]
    w_legvel = [0.01, 0.01, 0.01]
    w_x = np.diag(np.array(w_basepos + w_legpos * 4 + w_basevel + w_legvel * 4))

    w_linforce = np.array([0.01, 0.01, 0.01])
    w_u = np.concatenate(
        (
            w_linforce,
            w_linforce,
            w_linforce,
            w_linforce,
            np.ones(nv - 6) * 1e-5,
        )
    )
    w_u = np.diag(w_u)

    w_frame = np.diag(np.array([2000.0, 2000.0, 2000.0]))
    problem_conf = dict(
        timestep=record_config.dt_mpc,
        w_x=w_x,
        w_u=w_u,
        gravity=gravity,
        w_frame=w_frame,
        mu=0.8,
        kinematics_limits=False,
        land_cstr=True,
    )

    problem = QuadKinodynOcp(problem_conf, robot)
    problem.createProblem(robot.getReferenceState(), record_config.horizon, gravity[2])
    return problem


def build_recording_mpc(problem, robot, record_config: RecordConfig):
    gait = Gait(robot, record_config.gait_command, record_config.gait_period, record_config.dt_mpc)
    contact_phases, fly_steps, contact_steps = gait.build_cycle()
    mpc_conf = dict(
        TOL=record_config.mpc_tol,
        mu_init=record_config.mpc_mu_init,
        max_iters=record_config.mpc_max_iters,
        num_threads=record_config.mpc_num_threads,
        swing_apex=record_config.mpc_swing_apex,
        T_fly=fly_steps,
        T_contact=contact_steps,
        timestep=record_config.dt_mpc,
        rollout_timestep=record_config.dt_sim,
    )
    mpc = MPC(mpc_conf, problem)
    mpc.generateCycleHorizon(contact_phases)
    return mpc


def build_visualizer(record_config: RecordConfig) -> MPCMeshcatVisualizer:
    viewer_robot, robot = build_robot(record_config)
    return MPCMeshcatVisualizer(
        viewer_robot,
        robot,
        record_config.foot_names,
        record_config.dt_mpc,
        open_viewer=True,
        viewer_name=record_config.viewer_name,
    )


def build_recording_stack(record_config: RecordConfig):
    _viewer_robot, robot = build_robot(record_config)
    problem = build_kinodynamics_problem(robot, record_config)
    mpc = build_recording_mpc(problem, robot, record_config)
    return robot, mpc


def infer_reference_height(record_config: RecordConfig) -> float:
    _viewer_robot, robot = build_robot(record_config)
    return float(robot.getReferenceState()[2])


def main() -> None:
    record_config = RecordConfig()
    reference_height = infer_reference_height(record_config)
    record_config.apply_reference_height(reference_height)

    trajectory_generator = TrajectoryGenerator(record_config)
    yaml_recorder = YamlRecorder(
        target_dt=record_config.dt_sim,
        weight=record_config.target_weight,
    )
    commands = trajectory_generator.build_commands()
    visualizer = build_visualizer(record_config) if record_config.enable_visualization else None

    Path(record_config.recording_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"recording {len(commands)} command(s) to {record_config.recording_dir} "
        f"with dt={record_config.dt_mpc:.3f}s duration={record_config.target_duration:.2f}s"
    )

    outputs = []
    shared_robot = None
    shared_mpc = None
    x_current = None
    if not record_config.reset_each_round:
        shared_robot, shared_mpc = build_recording_stack(record_config)
        x_current = np.array(shared_robot.getReferenceState(), dtype=np.float64)

    for index, command_job in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] command={command_job.command} run={command_job.run_name}")
        if record_config.reset_each_round:
            robot, mpc = build_recording_stack(record_config)
            x_initial = np.array(robot.getReferenceState(), dtype=np.float64)
        else:
            robot, mpc = shared_robot, shared_mpc
            x_initial = x_current

        nq = robot.getModel().nq
        x_current = np.array(x_initial, dtype=np.float64)
        q_record = [x_current[:nq].copy()]
        v_record = [x_current[nq:].copy()]
        solve_times = []

        global_reference = trajectory_generator.build_target_trajectory(command_job.command, q0=q_record[0])
        if visualizer is not None:
            visualizer.display_offline_plan(global_reference["state_trajectory"])

        for step in range(record_config.mpc_loops):
            state_ref = trajectory_generator.build_mpc_reference_window(
                global_reference,
                start_index=step,
                horizon=record_config.horizon,
            )
            mpc.velocity_base = state_ref[0, nq : nq + 6]

            start_time = time.perf_counter()
            mpc.iterate(x_current, state_ref=state_ref)
            solve_time = time.perf_counter() - start_time
            solve_times.append(solve_time)

            x_current = mpc.rollout(x_current, mpc.us[0]).astype(np.float64, copy=False)
            q_record.append(x_current[:nq].copy())
            v_record.append(x_current[nq:].copy())

            if visualizer is not None:
                visualizer.display_configuration(x_current[:nq])
                visualizer.display_horizon(visualizer.capture_horizon(mpc))
                if record_config.visualization_sleep:
                    time.sleep(record_config.dt_mpc)

            if step % record_config.mpc_print_every == 0 or step == record_config.mpc_loops - 1:
                print(
                    f"[{command_job.run_name}] step={step:03d}/{record_config.mpc_loops - 1:03d} "
                    f"solve_time={solve_times[-1] * 1e3:.2f} ms pos={x_current[:3]}"
                )

        dataset = yaml_recorder.build_dataset(robot, q_record, v_record, source_dt=record_config.dt_mpc)
        output_path = yaml_recorder.save_dataset(
            dataset=dataset,
            output_dir=record_config.recording_dir,
            run_name=command_job.run_name,
        )
        outputs.append(
            {
                "output_path": output_path,
                "solve_times": np.asarray(solve_times, dtype=np.float64),
                "samples": len(q_record),
            }
        )

    solve_times = np.concatenate([item["solve_times"] for item in outputs]) if outputs else np.array([])
    mean_solve_time_ms = 0.0 if solve_times.size == 0 else float(solve_times.mean() * 1e3)
    print(
        "recording complete:",
        f"runs={len(outputs)}",
        f"mean_solve_time={mean_solve_time_ms:.2f} ms",
        f"output_dir={record_config.recording_dir}",
    )
    for item in outputs:
        print(f"saved {item['samples']} samples -> {item['output_path']}")


if __name__ == "__main__":
    main()
