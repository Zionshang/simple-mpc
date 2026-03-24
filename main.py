from __future__ import annotations

import argparse
import sys
import termios
import time
import tty

import example_robot_data as erd
import numpy as np

from mpc import Gait, MPC, MPCMeshcatVisualizer, QuadKinodynOcp, QuadRobot


BASE_JOINT_NAME = "root_joint"
FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DT_MPC = 0.01
HORIZON = 50
SIMULATION_STEPS = 500
GAIT_NAME = "trot"
GAIT_CYCLE_PERIOD = 0.6


def read_single_key():
    if not sys.stdin.isatty():
        raise RuntimeError("step debug mode requires a TTY")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def debug_play_step_by_step(visualizer, q_traj, horizons):
    print("Meshcat debug mode ready.")
    print("Press space to advance one step, 'q' to quit.")
    for step, (q, horizon) in enumerate(zip(q_traj, horizons)):
        visualizer.display_configuration(q)
        visualizer.display_horizon(horizon)
        print(f"step {step + 1}/{len(horizons)}", flush=True)
        while True:
            key = read_single_key()
            if key == " ":
                break
            if key.lower() == "q":
                return


def play_visualization(visualizer, q_traj, horizons):
    visualizer.play(q_traj, horizons, repeat=False)


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


def build_mpc(problem, robot, gravity, gait: Gait):
    contact_phases, fly_steps, contact_steps = gait.build_cycle()

    mpc_conf = dict(
        support_force=-robot.getMass() * gravity[2],
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

    velocity_base = np.zeros(6)
    velocity_base[0] = 1.0
    mpc.velocity_base = velocity_base
    return mpc


def rollout_ideal_mpc(mpc, x0, nq, visualizer):
    x_current = np.array(x0)
    q_traj = [np.array(x_current[:nq])]
    x_traj = [x_current]
    u_traj = []
    solve_times = []
    horizon_history = []

    for step in range(SIMULATION_STEPS):
        start = time.perf_counter()
        mpc.iterate(x_current)
        solve_times.append(time.perf_counter() - start)

        horizon_history.append(visualizer.capture_horizon(mpc))
        u_traj.append(np.array(mpc.us[0]))

        x_current = np.array(mpc.xs[1])
        x_traj.append(x_current)
        q_traj.append(np.array(x_current[:nq]))

        if step % 25 == 0:
            print(
                f"step={step:03d} solve_time={solve_times[-1] * 1e3:.2f} ms "
                f"body_pos={x_current[:3]}"
            )

    return {
        "q_traj": np.asarray(q_traj[:-1]),
        "x_traj": np.asarray(x_traj),
        "u_traj": np.asarray(u_traj),
        "solve_times": np.asarray(solve_times),
        "horizon_history": horizon_history,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Pure Python Go2 kinodynamics ideal simulation.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use keyboard-controlled step-by-step visualization instead of continuous playback.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    viewer_robot, robot = build_robot()
    gait = Gait(robot, GAIT_NAME, GAIT_CYCLE_PERIOD, DT_MPC)
    problem, gravity = build_kinodynamics_problem(robot)
    mpc = build_mpc(problem, robot, gravity, gait)
    visualizer = MPCMeshcatVisualizer(viewer_robot, robot, FOOT_NAMES, DT_MPC)

    x0 = np.array(robot.getReferenceState())
    rollout = rollout_ideal_mpc(mpc, x0, robot.getModel().nq, visualizer)

    print(
        "Ideal MPC rollout complete:",
        f"steps={len(rollout['u_traj'])}",
        f"mean_solve_time={rollout['solve_times'].mean() * 1e3:.2f} ms",
    )

    if args.debug:
        debug_play_step_by_step(visualizer, rollout["q_traj"], rollout["horizon_history"])
    else:
        play_visualization(visualizer, rollout["q_traj"], rollout["horizon_history"])


if __name__ == "__main__":
    main()
