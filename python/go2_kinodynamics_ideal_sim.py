from __future__ import annotations

import argparse
import sys
import termios
import time
import tty

import example_robot_data as erd
import numpy as np

from simple_mpc_py import MPC, MPCMeshcatVisualizer, QuadKinodynOcp, QuadRobot


BASE_JOINT_NAME = "root_joint"
FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DT_MPC = 0.01
HORIZON = 50
SIMULATION_STEPS = 500
T_DOUBLE_SUPPORT = 10
T_SINGLE_SUPPORT = 30


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
    w_legpos = [1, 1, 1]
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

    w_cent_lin = np.array([0.0, 0.0, 1.0])
    w_cent_ang = np.array([0.1, 0.1, 10.0])
    w_cent = np.diag(np.concatenate((w_cent_lin, w_cent_ang)))

    w_centder_lin = np.zeros(3)
    w_centder_ang = np.ones(3) * 0.1
    w_centder = np.diag(np.concatenate((w_centder_lin, w_centder_ang)))

    w_frame = np.diag(np.array([100.0, 100.0, 500.0]))
    problem_conf = dict(
        timestep=DT_MPC,
        w_x=w_x,
        w_u=w_u,
        w_cent=w_cent,
        w_centder=w_centder,
        gravity=gravity,
        w_frame=w_frame,
        qmin=robot.getModel().lowerPositionLimit[7:],
        qmax=robot.getModel().upperPositionLimit[7:],
        mu=0.8,
        kinematics_limits=False,
        land_cstr=False,
        cent_cost=False,
        centder_cost=False,
        term_cent_cost=False,
        term_dcm_cstr=False,
    )

    problem = QuadKinodynOcp(problem_conf, robot)
    problem.createProblem(
        robot.getReferenceState(),
        HORIZON,
        gravity[2],
        False,
    )
    return problem, gravity


def build_mpc(problem, robot, gravity):
    mpc_conf = dict(
        support_force=-robot.getMass() * gravity[2],
        TOL=1e-4,
        mu_init=1e-8,
        max_iters=1,
        num_threads=8,
        swing_apex=0.30,
        T_fly=T_SINGLE_SUPPORT,
        T_contact=T_DOUBLE_SUPPORT,
        timestep=DT_MPC,
    )
    mpc = MPC(mpc_conf, problem)

    contact_phase_quadru = {name: True for name in FOOT_NAMES}
    contact_phase_lift_fl_rr = {
        "FL_foot": False,
        "FR_foot": True,
        "RL_foot": True,
        "RR_foot": False,
    }
    contact_phase_lift_fr_rl = {
        "FL_foot": True,
        "FR_foot": False,
        "RL_foot": False,
        "RR_foot": True,
    }

    contact_phases = [contact_phase_quadru] * T_DOUBLE_SUPPORT
    contact_phases += [contact_phase_lift_fl_rr] * T_SINGLE_SUPPORT
    contact_phases += [contact_phase_quadru] * T_DOUBLE_SUPPORT
    contact_phases += [contact_phase_lift_fr_rl] * T_SINGLE_SUPPORT
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
                f"com={x_current[:3]}"
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
    problem, gravity = build_kinodynamics_problem(robot)
    mpc = build_mpc(problem, robot, gravity)
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
