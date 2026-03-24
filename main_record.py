from __future__ import annotations

import time
from pathlib import Path

import example_robot_data as erd
import numpy as np

from mpc import Gait, MPC, MPCMeshcatVisualizer, QuadKinodynOcp, QuadRobot
from record.config import RecordConfig
from record.trajectory import TrajectoryGenerator
from record.yaml_recording import YamlRecorder

BASE_JOINT_NAME = "root_joint"
FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DT_MPC = 0.01
DT_ROLLOUT = 0.005
HORIZON = 50
GAIT_NAME = "trot"
GAIT_CYCLE_PERIOD = 0.6

RECORD_DURATION = 4.0
OUTPUT_DIR = "results"
TARGET_DT = 0.005
TARGET_WEIGHT = 1.0
TARGET_VX = 1.0
TARGET_VY = 0.0
TARGET_VYAW = 0.0
TARGET_HEIGHT = None
ENABLE_BATCH_SCAN = True
BATCH_X_VALUES = [0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9]
BATCH_Y_VALUES = [0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6]
BATCH_YAW_VALUES = [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5]
ENABLE_VISUALIZATION = True
VISUALIZATION_SLEEP = False
RESET_EACH_ROUND = False


def build_robot():
    viewer_robot = erd.load("go2")
    robot = QuadRobot(viewer_robot.model, "standing", BASE_JOINT_NAME)
    for foot_name in FOOT_NAMES:
        robot.addPointFoot(foot_name, BASE_JOINT_NAME)
    return viewer_robot, robot


def build_kinodynamics_problem(robot):
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
        timestep=DT_MPC,
        w_x=w_x,
        w_u=w_u,
        gravity=gravity,
        w_frame=w_frame,
        mu=0.8,
        kinematics_limits=False,
        land_cstr=True,
    )

    problem = QuadKinodynOcp(problem_conf, robot)
    problem.createProblem(
        robot.getReferenceState(),
        HORIZON,
        gravity[2],
    )
    return problem, gravity


def build_recording_mpc(problem, robot, gravity, gait: Gait):
    contact_phases, fly_steps, contact_steps = gait.build_cycle()

    mpc_conf = dict(
        support_force=-robot.getMass() * gravity[2],
        TOL=1e-4,
        mu_init=1e-8,
        max_iters=2,
        num_threads=8,
        swing_apex=0.30,
        T_fly=fly_steps,
        T_contact=contact_steps,
        timestep=DT_MPC,
        rollout_timestep=DT_ROLLOUT,
    )
    mpc = MPC(mpc_conf, problem)
    mpc.generateCycleHorizon(contact_phases)
    return mpc


def infer_reference_height() -> float:
    _viewer_robot, robot = build_robot()
    return float(robot.getReferenceState()[2])


def make_record_config(reference_height: float) -> RecordConfig:
    target_height = reference_height if TARGET_HEIGHT is None else float(TARGET_HEIGHT)
    return RecordConfig(
        target_velocity_command=np.array([TARGET_VX, TARGET_VY, TARGET_VYAW, target_height], dtype=np.float64),
        dt=DT_MPC,
        target_duration=float(RECORD_DURATION),
        gait_command=GAIT_NAME,
        gait_period=GAIT_CYCLE_PERIOD,
        batch_scan_mode=bool(ENABLE_BATCH_SCAN),
        batch_scan_height=target_height,
        batch_scan_x_values=list(BATCH_X_VALUES),
        batch_scan_y_values=list(BATCH_Y_VALUES),
        batch_scan_yaw_values=list(BATCH_YAW_VALUES),
        recording_dir=OUTPUT_DIR,
    )


def build_visualizer() -> MPCMeshcatVisualizer:
    viewer_robot, robot = build_robot()
    return MPCMeshcatVisualizer(
        viewer_robot,
        robot,
        FOOT_NAMES,
        DT_MPC,
        open_viewer=True,
        viewer_name="simple_mpc_record",
    )


def build_recording_stack():
    _viewer_robot, robot = build_robot()
    gait = Gait(robot, GAIT_NAME, GAIT_CYCLE_PERIOD, DT_MPC)
    problem, gravity = build_kinodynamics_problem(robot)
    mpc = build_recording_mpc(problem, robot, gravity, gait)
    return robot, mpc


def rollout_command(
    command_job,
    record_config: RecordConfig,
    trajectory_generator: TrajectoryGenerator,
    yaml_recorder: YamlRecorder,
    visualizer: MPCMeshcatVisualizer | None,
    sleep: bool,
    robot,
    mpc: MPC,
    x_initial: np.ndarray,
) -> tuple[dict, np.ndarray]:
    nq = robot.getModel().nq

    x_current = np.array(x_initial, dtype=np.float64)
    q_record = [x_current[:nq].copy()]
    v_record = [x_current[nq:].copy()]
    solve_times = []

    offline_reference = trajectory_generator.build_target_trajectory(command_job.command, q0=q_record[0])
    if visualizer is not None:
        visualizer.display_offline_plan(offline_reference["state_trajectory"])

    for step in range(record_config.mpc_loops):
        state_reference = trajectory_generator.build_mpc_reference_window(
            offline_reference,
            start_index=step,
            horizon=HORIZON,
        )
        mpc.velocity_base = state_reference[0, nq : nq + 6]

        start = time.perf_counter()
        mpc.iterate(x_current, state_ref=state_reference)
        solve_times.append(time.perf_counter() - start)

        x_current = mpc.rollout(x_current, mpc.us[0]).astype(np.float64, copy=False)
        q_record.append(x_current[:nq].copy())
        v_record.append(x_current[nq:].copy())

        if visualizer is not None:
            visualizer.display_configuration(q_record[-1])
            visualizer.display_horizon(visualizer.capture_horizon(mpc))
            if sleep:
                time.sleep(DT_MPC)

        if step % 25 == 0 or step == record_config.mpc_loops - 1:
            print(
                f"[{command_job.run_name}] step={step:03d}/{record_config.mpc_loops - 1:03d} "
                f"solve_time={solve_times[-1] * 1e3:.2f} ms pos={x_current[:3]}"
            )

    dataset = yaml_recorder.build_dataset(robot, q_record, v_record, source_dt=DT_MPC)
    output_path = yaml_recorder.save_dataset(
        dataset=dataset,
        output_dir=record_config.recording_dir,
        run_name=command_job.run_name,
    )
    return {
        "output_path": output_path,
        "solve_times": np.asarray(solve_times, dtype=np.float64),
        "samples": len(q_record),
    }, x_current.copy()


def main() -> None:
    reference_height = infer_reference_height()
    record_config = make_record_config(reference_height=reference_height)
    trajectory_generator = TrajectoryGenerator(record_config)
    yaml_recorder = YamlRecorder(target_dt=TARGET_DT, weight=TARGET_WEIGHT)
    commands = trajectory_generator.build_commands()
    visualizer = build_visualizer() if ENABLE_VISUALIZATION else None

    Path(record_config.recording_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"recording {len(commands)} command(s) to {record_config.recording_dir} "
        f"with dt={DT_MPC:.3f}s duration={record_config.target_duration:.2f}s"
    )

    outputs = []
    shared_robot = None
    shared_mpc = None
    x_current = None
    if not RESET_EACH_ROUND:
        shared_robot, shared_mpc = build_recording_stack()
        x_current = np.array(shared_robot.getReferenceState(), dtype=np.float64)

    for index, command_job in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] command={command_job.command} run={command_job.run_name}")
        if RESET_EACH_ROUND:
            robot, mpc = build_recording_stack()
            x_initial = np.array(robot.getReferenceState(), dtype=np.float64)
        else:
            robot, mpc = shared_robot, shared_mpc
            x_initial = x_current

        output, x_current = rollout_command(
            command_job,
            record_config,
            trajectory_generator,
            yaml_recorder,
            visualizer=visualizer,
            sleep=VISUALIZATION_SLEEP,
            robot=robot,
            mpc=mpc,
            x_initial=x_initial,
        )
        outputs.append(output)

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
