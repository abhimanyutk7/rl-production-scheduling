from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .config import ShopConfig
from .instances import generate_jobs, load_instance


@dataclass
class Machine:
    power: float
    health: float = 1.0
    mode: str = "idle"
    remaining: int = 0
    job: int = -1
    product: int = -1
    continuous: int = 0
    processing_left: int = 0
    resume_mode: str = "process"
    resume_left: int = 0
    last_maintenance: int = 0


class FactoryEnv(gym.Env):
    """Event decisions with one-minute physical integration and fixed demand windows."""
    metadata = {"render_modes": []}

    def __init__(self, config=None, objective="joint", instance=None, record=False, strict_actions=False):
        super().__init__()
        self.config = config or ShopConfig()
        if objective not in ("joint", "wear", "peak", "lateness"):
            raise ValueError("Unknown objective")
        self.objective, self.instance, self.record, self.strict_actions = objective, instance, record, strict_actions
        c = self.config
        self.maintenance_action = c.shortlist
        self.action_space = spaces.Discrete(c.shortlist + 1)
        self.obs_size = 13 + c.shortlist * (7 + c.products) + c.machine_slots * (9 + c.products)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_size,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.config
        self.jobs = load_instance(self.instance, c) if self.instance else generate_jobs(c, self.np_random, seed)
        self.machines = [Machine(float(10 + 5 * (i % 5))) for i in range(c.machines)]
        self.operation = np.zeros(c.jobs, dtype=int)
        self.active = np.zeros(c.jobs, dtype=bool)
        self.completion = np.full(c.jobs, -1, dtype=int)
        self.time = self.steps = self.invalid_actions = self.breakdowns = self.maintenance_count = 0
        self.energy_kwh = self.energy_cost = self.window_kwh = self.tardiness = self.maintenance_downtime = self.repair_downtime = 0.0
        self.initial_peak = float(self.np_random.uniform(20, 50)) if c.random_initial_peak else c.initial_peak_kw
        self.peak_kw = self.initial_peak
        self.shift_peak_kw = 0.0
        self.health_integral = 0.0
        self.productive_minutes = np.zeros(c.machines)
        self.reward_total = 0.0
        self.terminated = self.truncated = False
        self.trace, self.events, self.demand_windows = [], [], []
        # Common machine/time shocks keep comparisons paired without action-dependent RNG consumption.
        self.shocks = self.np_random.random((c.shift_minutes, c.machines))
        self.current_machine = -1
        self._seek_decision()
        return self._observation(), self._info()

    def eligible(self, machine):
        return [j for j, job in enumerate(self.jobs) if job.arrival <= self.time and self.completion[j] < 0 and not self.active[j]
                and machine in job.operations[self.operation[j]]]

    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        candidates = self.candidates()
        mask[:len(candidates)] = True
        if self.current_machine >= 0 and not (self.terminated or self.truncated):
            machine = self.machines[self.current_machine]
            mask[self.maintenance_action] = self.config.maintenance_enabled and machine.health < 0.98
        if not mask.any():
            mask[0] = True
        return mask

    def candidates(self):
        if self.current_machine < 0 or self.terminated or self.truncated:
            return []
        return sorted(self.eligible(self.current_machine), key=lambda j: (self.jobs[j].due, self.jobs[j].arrival, j))[:self.config.shortlist]

    def processing_power(self, machine):
        product = self.jobs[machine.job].product if machine.job >= 0 else 0
        return machine.power * (0.8 + 0.2 * product)

    def setup_time(self, machine, job):
        if not self.config.setup_enabled or machine.product in (-1, job.product):
            return 0
        return 5 + 3 * ((job.product - machine.product) % self.config.products)

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("Reset an ended episode before stepping")
        before = self.cost_components()
        raw = np.asarray(action)
        valid_scalar = raw.ndim == 0 and raw.dtype.kind in "iu"
        action = int(raw) if valid_scalar else -1
        if not 0 <= action < self.action_space.n or not self.action_masks()[action]:
            self.invalid_actions += 1
            if self.strict_actions:
                raise ValueError("Action is not legal in this state")
            action = 0
        m = self.current_machine
        machine = self.machines[m]
        if action < self.config.shortlist:
            j = self.candidates()[action]
            job = self.jobs[j]
            setup = self.setup_time(machine, job)
            machine.job = j
            machine.processing_left = job.operations[self.operation[j]][m]
            machine.mode = "setup" if setup else "process"
            machine.remaining = setup if setup else machine.processing_left
            self.active[j] = True
        elif action == self.maintenance_action:
            machine.mode, machine.remaining = "maintenance", self.config.maintenance_minutes
            self.maintenance_count += 1
        if self.record:
            self.events.append({"time": self.time, "machine": m, "action": action, "health": machine.health})
        self.steps += 1
        self._seek_decision()
        after = self.cost_components()
        if self.objective == "joint":
            delta = after["full"] - before["full"]
        else:
            delta = after["lateness"] - before["lateness"]
            if self.objective == "wear":
                delta += after["maintenance"] - before["maintenance"]
            elif self.objective == "peak":
                delta += after["energy"] + after["demand"] - before["energy"] - before["demand"]
        reward = -delta / self.config.reward_scale
        self.reward_total += reward
        return self._observation(), float(reward), self.terminated, self.truncated, self._info()

    def _seek_decision(self):
        while True:
            self.terminated = bool(np.all(self.completion >= 0))
            self.truncated = self.time >= self.config.shift_minutes and not self.terminated
            if self.terminated or self.truncated:
                self.current_machine = -1
                return
            for m, machine in enumerate(self.machines):
                if machine.mode == "idle" and self.eligible(m):
                    self.current_machine = m
                    return
            self._tick()

    def _tick(self):
        c = self.config
        power = 0.0
        for m, machine in enumerate(self.machines):
            mode = machine.mode
            kw = self.processing_power(machine) if mode == "process" else machine.power * 0.4 if mode == "setup" else 0.8 if mode == "idle" else 0.2
            power += kw
            if self.record:
                self.trace.append({"time": self.time, "machine": m, "mode": mode, "job": machine.job,
                                   "operation": int(self.operation[machine.job]) if machine.job >= 0 else -1, "power_kw": kw, "health": machine.health})
            if mode == "idle":
                machine.continuous = 0
                continue
            machine.remaining -= 1
            if mode in ("process", "setup"):
                machine.continuous += 1
                machine.health = max(0.0, machine.health - c.wear_strength * (c.wear_per_minute * kw / 20 + c.continuous_wear * machine.continuous / 120))
                if mode == "process":
                    self.productive_minutes[m] += 1
                failure_p = c.failure_base + (c.failure_health * (1 - machine.health) if c.load_dependent_failure else c.failure_health * 0.25)
                if machine.remaining > 0 and self.shocks[self.time, m] < failure_p:
                    machine.resume_mode, machine.resume_left = mode, machine.remaining
                    machine.mode, machine.remaining = "repair", c.repair_minutes
                    self.breakdowns += 1
                    if self.record:
                        self.events.append({"time": self.time + 1, "machine": m, "event": "breakdown"})
                    continue
            elif mode == "maintenance":
                self.maintenance_downtime += 1
            elif mode == "repair":
                self.repair_downtime += 1
            if machine.remaining == 0:
                if mode == "setup":
                    machine.mode, machine.remaining = "process", machine.processing_left
                elif mode == "process":
                    j = machine.job
                    machine.product = self.jobs[j].product
                    self.operation[j] += 1
                    self.active[j] = False
                    if self.operation[j] == len(self.jobs[j].operations):
                        self.completion[j] = self.time + 1
                    machine.mode, machine.job = "idle", -1
                elif mode == "maintenance":
                    machine.health, machine.continuous, machine.mode = 1.0, 0, "idle"
                    machine.last_maintenance = self.time + 1
                elif mode == "repair":
                    machine.health, machine.continuous = 0.8, 0
                    machine.mode, machine.remaining = machine.resume_mode, machine.resume_left
        kwh = power / 60
        self.energy_kwh += kwh
        self.energy_cost += kwh * c.tariff(self.time)
        self.window_kwh += kwh
        window_average = self.window_kwh * 60 / c.demand_window
        self.shift_peak_kw = max(self.shift_peak_kw, window_average)
        self.peak_kw = max(self.peak_kw, window_average)
        self.tardiness += sum(job.due <= self.time and self.completion[j] < 0 for j, job in enumerate(self.jobs))
        # Completion at t+1 also incurs its final overdue minute.
        self.tardiness += sum(job.due <= self.time and self.completion[j] == self.time + 1 for j, job in enumerate(self.jobs))
        self.health_integral += sum(m.health for m in self.machines)
        self.time += 1
        if self.record and self.trace:
            self.trace[-1]["total_power_kw"] = power
        if self.time % c.demand_window == 0:
            self.demand_windows.append({"end_minute": self.time, "average_kw": window_average})
            self.window_kwh = 0.0

    def cost_components(self):
        c = self.config
        lateness = c.tardiness_weight * self.tardiness / c.tardiness_scale
        if self.truncated:
            lateness += c.unfinished_penalty * int(np.sum(self.completion < 0))
        energy = c.energy_weight * self.energy_cost / c.energy_scale
        demand = c.demand_weight * c.demand_rate * (self.peak_kw - self.initial_peak) / c.demand_scale if c.demand_enabled else 0.0
        maintenance = c.maintenance_weight * (self.maintenance_downtime + self.repair_downtime) / c.downtime_scale
        return {"lateness": lateness, "energy": energy, "demand": demand, "maintenance": maintenance,
                "full": lateness + energy + demand + maintenance}

    def _observation(self):
        c = self.config
        clock = 2 * np.pi * ((c.start_hour * 60 + self.time) % 1440) / 1440
        selected = self.machines[self.current_machine] if self.current_machine >= 0 else None
        waiting = sum(job.arrival <= self.time and self.completion[j] < 0 and not self.active[j] for j, job in enumerate(self.jobs))
        late = sum(job.arrival <= self.time and job.due < self.time and self.completion[j] < 0 for j, job in enumerate(self.jobs))
        obs = [self.time / c.shift_minutes, np.sin(clock), np.cos(clock), c.tariff(self.time) / max(1, c.peak_rate), self.peak_kw / 300,
               self.window_kwh * 60 / c.demand_window / 300, (self.time % c.demand_window) / c.demand_window,
               waiting / c.jobs, late / c.jobs, 1 - self.time / c.shift_minutes,
               selected.health if selected and c.health_visible else 0.0,
               (self.time - selected.last_maintenance) / c.shift_minutes if selected else 0.0,
               selected.continuous / c.shift_minutes if selected else 0.0]
        candidates = self.candidates()
        for slot in range(c.shortlist):
            if slot >= len(candidates):
                obs.extend([0.0] * (7 + c.products))
                continue
            j = candidates[slot]
            job, op = self.jobs[j], self.operation[j]
            remaining_work = sum(min(operation.values()) for operation in job.operations[op:])
            obs.extend([1.0, job.operations[op][self.current_machine] / 60,
                        (job.due - self.time - remaining_work) / c.shift_minutes,
                        self.setup_time(selected, job) / 30, (len(job.operations) - op) / c.max_operations,
                        selected.power * (0.8 + 0.2 * job.product) / 50, (self.time - job.arrival) / c.shift_minutes,
                        *[float(job.product == p) for p in range(c.products)]])
        for m in range(c.machine_slots):
            if m >= c.machines:
                obs.extend([0.0] * (9 + c.products))
                continue
            machine = self.machines[m]
            obs.extend([1.0, float(m == self.current_machine), machine.health if c.health_visible else 0.0,
                        float(machine.mode != "idle"), machine.remaining / c.shift_minutes,
                        len(self.eligible(m)) / c.jobs, machine.power / 50, machine.continuous / c.shift_minutes,
                        (self.time - machine.last_maintenance) / c.shift_minutes,
                        *[float(machine.product == p) for p in range(c.products)]])
        return np.asarray(obs, dtype=np.float32)

    def _info(self):
        return {"time": self.time, "machine": self.current_machine, "action_mask": self.action_masks(), "metrics": self.metrics(), "cost_components": self.cost_components()}

    def metrics(self):
        c = self.config
        done = self.completion >= 0
        late = [max(0, int(t if t >= 0 else self.time) - j.due) for j, t in zip(self.jobs, self.completion)]
        incremental_demand = c.demand_rate * (self.peak_kw - self.initial_peak) if c.demand_enabled else 0.0
        full_demand = c.demand_rate * self.peak_kw if c.demand_enabled else 0.0
        return {"makespan_min": self.time, "overtime_min": max(0, self.time - c.shift_minutes), "completed_jobs": int(sum(done)),
                "unfinished_jobs": int(sum(~done)), "completion_rate": float(np.mean(done)), "average_tardiness_min": float(np.mean(late)) if late else 0.0,
                "worst_tardiness_min": max(late, default=0), "late_jobs": sum(x > 0 for x in late),
                "breakdowns": self.breakdowns, "maintenance_count": self.maintenance_count, "maintenance_downtime_min": self.maintenance_downtime,
                "repair_downtime_min": self.repair_downtime, "final_average_health": float(np.mean([m.health for m in self.machines])),
                "utilisation": float(sum(self.productive_minutes) / max(1, self.time * c.machines)),
                "average_health": self.health_integral / max(1, self.time * c.machines), "energy_kwh": self.energy_kwh,
                "initial_peak_kw": self.initial_peak, "shift_peak_kw": self.shift_peak_kw, "billing_peak_kw": self.peak_kw,
                "energy_cost": self.energy_cost, "incremental_demand_cost": incremental_demand,
                "incremental_electricity_cost": self.energy_cost + incremental_demand,
                "total_electricity_bill": self.energy_cost + full_demand, "full_objective": self.cost_components()["full"],
                "invalid_actions": self.invalid_actions, "decisions": self.steps, "truncated": self.truncated}
