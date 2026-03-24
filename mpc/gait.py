from __future__ import annotations

from dataclasses import dataclass

from .robot_handler import QuadRobot


@dataclass(frozen=True)
class Gait:
    robot: QuadRobot
    name: str
    cycle_period: float
    timestep: float

    def build_cycle(self) -> tuple[list[dict[str, bool]], int, int]:
        cycle_steps = int(round(self.cycle_period / self.timestep))
        if cycle_steps <= 0:
            raise ValueError("cycle_period must be positive")

        fl, fr, rl, rr = self.robot.getFeetFrameNames()
        contact_phase_quadru = {name: True for name in self.robot.getFeetFrameNames()}
        contact_phase_lift_fl_rr = {
            fl: False,
            fr: True,
            rl: True,
            rr: False,
        }
        contact_phase_lift_fr_rl = {
            fl: True,
            fr: False,
            rl: False,
            rr: True,
        }

        if self.name == "trot":
            if cycle_steps < 2:
                raise ValueError("trot needs at least 2 steps per cycle")
            phase_1_steps = cycle_steps // 2
            phase_2_steps = cycle_steps - phase_1_steps
            contact_phases = [contact_phase_lift_fl_rr] * phase_1_steps
            contact_phases += [contact_phase_lift_fr_rl] * phase_2_steps
            return contact_phases, phase_1_steps, 0

        if self.name == "stand_trot":
            if cycle_steps < 4:
                raise ValueError("stand_trot needs at least 4 steps per cycle")
            half_cycle_steps = cycle_steps // 2
            other_half_cycle_steps = cycle_steps - half_cycle_steps

            contact_steps = max(1, int(round(0.25 * half_cycle_steps)))
            contact_steps = min(contact_steps, half_cycle_steps - 1)
            fly_steps = half_cycle_steps - contact_steps

            other_contact_steps = max(1, int(round(0.25 * other_half_cycle_steps)))
            other_contact_steps = min(other_contact_steps, other_half_cycle_steps - 1)
            other_fly_steps = other_half_cycle_steps - other_contact_steps

            contact_phases = [contact_phase_quadru] * contact_steps
            contact_phases += [contact_phase_lift_fl_rr] * fly_steps
            contact_phases += [contact_phase_quadru] * other_contact_steps
            contact_phases += [contact_phase_lift_fr_rl] * other_fly_steps
            return contact_phases, fly_steps, contact_steps

        raise ValueError(f"Unsupported gait: {self.name}")
