from __future__ import annotations

import time

import example_robot_data as erd
import numpy as np

from mpc import Gait, MPC, MPCMeshcatVisualizer, QuadKinodynOcp, QuadRobot, StatePlanner


BASE_JOINT_NAME = "root_joint"
FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DT_MPC = 0.01
DT_ROLLOUT = 0.005
HORIZON = 50
SIMULATION_STEPS = 500
GAIT_NAME = "trot"
GAIT_CYCLE_PERIOD = 0.6


def build_robot():
    viewer_robot = erd.load("go2")
    robot = QuadRobot(viewer_robot.model, "standing", BASE_JOINT_NAME)
    for foot_name in FOOT_NAMES:
        robot.addPointFoot(foot_name, BASE_JOINT_NAME)
    return viewer_robot, robot


def build_kinodynamics_problem(robot):
    gravity = np.array([0.0, 0.0, -9.81])
    nv = robot.getModel().nv
    w_basepos = [0, 0, 100, 10, 10, 0]
    w_legpos = [5, 1, 1]
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

    w_frame = np.diag(np.array([100.0, 100.0, 500.0]))
    problem_conf = dict(
        timestep=DT_MPC,
        w_x=w_x,
        w_u=w_u,
        gravity=gravity,
        w_frame=w_frame,
        mu=0.8,
        kinematics_limits=False,
        land_cstr=False,
    )

    problem = QuadKinodynOcp(problem_conf, robot)
    problem.createProblem(
        robot.getReferenceState(),
        HORIZON,
        gravity[2],
    )
    return problem, gravity


def build_state_ref(
    state_planner: StatePlanner,
    x_current: np.ndarray,
    velocity_base: np.ndarray,
) -> np.ndarray:
    state_planner.updateReference(x_current, velocity_base)
    return np.asarray(
        [state_planner.getReference(t) for t in range(HORIZON)],
        dtype=float,
    )


def build_mpc(problem, robot, gravity, gait: Gait):
    contact_phases, fly_steps, contact_steps = gait.build_cycle()

    mpc_conf = dict(
        TOL=1e-4,
        mu_init=1e-8,
        max_iters=1,
        num_threads=8,
        swing_apex=0.30,
        T_fly=fly_steps,
        T_contact=contact_steps,
        timestep=DT_MPC,
    )
    mpc = MPC(mpc_conf, problem)
    mpc.generateCycleHorizon(contact_phases)
    return mpc


def main():
    viewer_robot, robot = build_robot()
    gait = Gait(robot, GAIT_NAME, GAIT_CYCLE_PERIOD, DT_MPC)
    problem, gravity = build_kinodynamics_problem(robot)
    mpc = build_mpc(problem, robot, gravity, gait)
    state_planner = StatePlanner(robot, HORIZON, DT_MPC)
    visualizer = MPCMeshcatVisualizer(viewer_robot, robot, FOOT_NAMES, DT_ROLLOUT)

    velocity_base = np.zeros(6)
    velocity_base[0] = 1.0
    rollout_steps_per_mpc = int(round(DT_MPC / DT_ROLLOUT))
    x_current = np.array(robot.getReferenceState())
    q_traj = [np.array(x_current[: robot.getModel().nq])]
    x_traj = [x_current]
    u_traj = []
    solve_times = []
    horizon_history = []

    for step in range(SIMULATION_STEPS):
        state_reference = build_state_ref(
            state_planner,
            x_current,
            velocity_base,
        )

        start = time.perf_counter()
        mpc.iterate(x_current, state_reference)
        solve_times.append(time.perf_counter() - start)

        horizon = visualizer.capture_horizon(mpc)
        u0 = np.array(mpc.us[0])
        u_traj.append(u0)

        for _ in range(rollout_steps_per_mpc):
            x_current = mpc.rollout(x_current, u0, DT_ROLLOUT)
            x_traj.append(x_current)
            q_traj.append(np.array(x_current[: robot.getModel().nq]))
            horizon_history.append(horizon)

        if step % 25 == 0:
            print(
                f"step={step:03d} solve_time={solve_times[-1] * 1e3:.2f} ms "
                f"body_pos={x_current[:3]}"
            )

    rollout = {
        "q_traj": np.asarray(q_traj[:-1]),
        "x_traj": np.asarray(x_traj),
        "u_traj": np.asarray(u_traj),
        "solve_times": np.asarray(solve_times),
        "horizon_history": horizon_history,
    }

    print(
        "Ideal MPC rollout complete:",
        f"steps={len(rollout['u_traj'])}",
        f"mean_solve_time={rollout['solve_times'].mean() * 1e3:.2f} ms",
    )

    visualizer.play(rollout["q_traj"], rollout["horizon_history"], repeat=False)


if __name__ == "__main__":
    main()
