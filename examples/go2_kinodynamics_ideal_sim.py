import time

import example_robot_data as erd
import numpy as np

from simple_mpc import KinodynamicsOCP, MPC, RobotModelHandler
from utils import MPCMeshcatVisualizer


BASE_JOINT_NAME = "root_joint"
FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DT_MPC = 0.01
HORIZON = 50
SIMULATION_STEPS = 500
T_DOUBLE_SUPPORT = 10
T_SINGLE_SUPPORT = 30


def build_model_handler():
    robot = erd.load("go2")
    model_handler = RobotModelHandler(robot.model, "standing", BASE_JOINT_NAME)
    for foot_name in FOOT_NAMES:
        model_handler.addPointFoot(foot_name, BASE_JOINT_NAME)
    return robot, model_handler


def build_kinodynamics_problem(model_handler):
    gravity = np.array([0.0, 0.0, -9.81])
    nv = model_handler.getModel().nv
    force_size = 3

    w_basepos = [0, 0, 100, 10, 10, 0]
    w_legpos = [1, 1, 1]
    w_basevel = [10, 10, 10, 10, 10, 10]
    w_legvel = [0.1, 0.1, 0.1]
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

    problem_conf = dict(
        timestep=DT_MPC,
        w_x=w_x,
        w_u=w_u,
        w_cent=w_cent,
        w_centder=w_centder,
        gravity=gravity,
        force_size=force_size,
        w_frame=np.eye(3) * 2000.0,
        qmin=model_handler.getModel().lowerPositionLimit[7:],
        qmax=model_handler.getModel().upperPositionLimit[7:],
        mu=0.8,
        Lfoot=0.01,
        Wfoot=0.01,
        kinematics_limits=True,
        force_cone=False,
        land_cstr=False,
    )

    problem = KinodynamicsOCP(problem_conf, model_handler)
    problem.createProblem(
        model_handler.getReferenceState(),
        HORIZON,
        force_size,
        gravity[2],
        False,
    )
    return problem, gravity


def build_mpc(problem, model_handler, gravity):
    mpc_conf = dict(
        support_force=-model_handler.getMass() * gravity[2],
        TOL=1e-4,
        mu_init=1e-8,
        max_iters=1,
        num_threads=8,
        swing_apex=0.15,
        T_fly=T_SINGLE_SUPPORT,
        T_contact=T_DOUBLE_SUPPORT,
        timestep=DT_MPC,
    )
    mpc = MPC(mpc_conf, problem)

    contact_phase_quadru = {
        "FL_foot": True,
        "FR_foot": True,
        "RL_foot": True,
        "RR_foot": True,
    }
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
    x_current = x0.copy()
    q_traj = [x_current[:nq].copy()]
    x_traj = [x_current.copy()]
    u_traj = []
    solve_times = []
    horizon_history = []

    for step in range(SIMULATION_STEPS):
        start = time.perf_counter()
        mpc.iterate(x_current)
        solve_times.append(time.perf_counter() - start)

        horizon_history.append(visualizer.capture_horizon(mpc))
        u_traj.append(np.array(mpc.us[0]).copy())

        x_current = np.array(mpc.xs[1]).copy()
        x_traj.append(x_current.copy())
        q_traj.append(x_current[:nq].copy())

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


def main():
    robot, model_handler = build_model_handler()
    problem, gravity = build_kinodynamics_problem(model_handler)
    mpc = build_mpc(problem, model_handler, gravity)
    visualizer = MPCMeshcatVisualizer(robot, model_handler, FOOT_NAMES, DT_MPC)

    x0 = np.array(model_handler.getReferenceState()).copy()
    rollout = rollout_ideal_mpc(mpc, x0, model_handler.getModel().nq, visualizer)

    print(
        "Ideal MPC rollout complete:",
        f"steps={len(rollout['u_traj'])}",
        f"mean_solve_time={rollout['solve_times'].mean() * 1e3:.2f} ms",
    )

    visualizer.play(rollout["q_traj"], rollout["horizon_history"], repeat=True)


if __name__ == "__main__":
    main()
