"""Persistent, partially observed production scheduling with costly inspections."""
from dataclasses import dataclass
import copy
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import ShopConfig
from .env import Machine
from .instances import generate_jobs
from .belief import HealthBelief


@dataclass(frozen=True)
class InspectionConfig(ShopConfig):
    jobs: int = 30  # Orders per operating day, not total orders.
    start_hour: int = 14
    days: int = 60
    billing_days: int = 30
    particles: int = 256
    inspection_minutes: int = 5
    inspection_noise: float = 0.03
    inspection_cost: float = 100.0
    control_minutes: int = 10
    wait_minutes: int = 5
    maintenance_cooldown: int = 30
    initial_health_low: float = 0.6
    wear_rate_sigma: float = 0.2
    wear_noise_cv: float = 0.35
    inspections_enabled: bool = True
    random_initial_peak: bool = False
    initial_peak_kw: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        for name in ('days', 'billing_days', 'particles', 'inspection_minutes',
                     'control_minutes', 'wait_minutes', 'maintenance_cooldown'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.particles < 16:
            raise ValueError('Use at least 16 condition particles')
        if self.start_hour * 60 + self.shift_minutes > 1440:
            raise ValueError('Operating shifts must end within their calendar day')
        if 1440 % self.demand_window:
            raise ValueError('Demand windows must divide a calendar day')
        if not 0 <= self.initial_health_low <= 1:
            raise ValueError('initial_health_low must be in [0,1]')
        if type(self.inspections_enabled) is not bool:
            raise ValueError('inspections_enabled must be boolean')
        for name in ('inspection_noise', 'inspection_cost', 'wear_rate_sigma', 'wear_noise_cv'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative')

    @property
    def horizon(self):
        return (self.days - 1) * 1440 + self.shift_minutes


@dataclass
class InspectionMachine(Machine):
    defer_until: int = 0
    next_control: int = 0
    pm_ready: int = 0
    inspection_ready: int = 0
    last_measurement: int = -1


GLOBAL_FEATURES = ('elapsed', 'clock_sin', 'clock_cos', 'tariff', 'billing_peak',
                   'window_average', 'window_fraction', 'waiting', 'late',
                   'shift_left', 'episode_left', 'billing_day', 'periods_left',
                   'selected_machine', 'inspection_enabled', 'inspection_cost', 'inspection_noise')
MACHINE_FEATURES = ('present', 'selected', 'health_mean', 'health_std', 'health_q05',
                    'health_q95', 'busy', 'remaining', 'eligible', 'power', 'continuous',
                    'maintenance_age', 'measurement_age', 'measured', 'deferred', 'pm_cooldown')
JOB_FEATURES = ('present', 'duration', 'slack', 'setup', 'operations_left', 'power', 'age')


class InspectionEnv(gym.Env):
    """Finite multi-day POMDP simulator; ordinary controllers use policy_state()."""
    metadata = {'render_modes': []}
    schema_version = 'inspection-v1'

    def __init__(self, config=None, record=False, exact_health=False, inspections_enabled=True, strict_actions=False):
        super().__init__()
        self.config = config or InspectionConfig()
        self.record, self.exact_health = record, exact_health
        self.strict_actions = strict_actions
        self.inspections_enabled = inspections_enabled and self.config.inspections_enabled
        c = self.config
        self.maintenance_action, self.inspection_action, self.wait_action = c.shortlist, c.shortlist + 1, c.shortlist + 2
        self.action_space = spaces.Discrete(c.shortlist + 3)
        self.observation_schema = list(GLOBAL_FEATURES)
        for j in range(c.shortlist):
            self.observation_schema += [f'job{j}.{name}' for name in JOB_FEATURES]
            self.observation_schema += [f'job{j}.product{p}' for p in range(c.products)]
        for m in range(c.machine_slots):
            self.observation_schema += [f'machine{m}.{name}' for name in MACHINE_FEATURES]
            self.observation_schema += [f'machine{m}.product{p}' for p in range(c.products)]
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(self.observation_schema),), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.config
        self.scenario_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        self.time = self.open_minutes = self.steps = self.invalid_actions = 0
        self.current_machine = -1
        self.terminated = False
        self.jobs = []
        for day in range(c.days):
            rng = np.random.default_rng(np.random.SeedSequence([self.scenario_seed, 101, day]))
            for _ in range(10000):
                daily = generate_jobs(c, rng, self.scenario_seed)
                if daily[-1].arrival < c.shift_minutes:
                    break
            else:
                raise ValueError('Arrival stream cannot fit operating shift; reduce arrival_minutes')
            for job in daily:
                job.arrival += day * 1440
                job.due += day * 1440
            self.jobs.extend(daily)
        self.arrivals = np.array([j.arrival for j in self.jobs])
        self.dues = np.array([j.due for j in self.jobs])
        self.operation = np.zeros(len(self.jobs), dtype=int)
        self.active = np.zeros(len(self.jobs), dtype=bool)
        self.completion = np.full(len(self.jobs), -1, dtype=int)
        physical = np.random.default_rng(np.random.SeedSequence([self.scenario_seed, 201]))
        self.machines = [InspectionMachine(float(10 + 5 * (m % 5)),
                         health=float(physical.uniform(c.initial_health_low, 1))) for m in range(c.machines)]
        self.true_rates = physical.lognormal(-c.wear_rate_sigma**2/2, c.wear_rate_sigma, c.machines)
        shape = (c.horizon, c.machines)
        self.failure_shocks = physical.random(shape)
        self.wear_shocks = physical.gamma(1 / c.wear_noise_cv**2, c.wear_noise_cv**2, shape) if c.wear_noise_cv else np.ones(shape)
        sensor = np.random.default_rng(np.random.SeedSequence([self.scenario_seed, 301]))
        self.sensor_shocks = sensor.normal(size=shape)
        self.beliefs = [HealthBelief(c, np.random.default_rng(np.random.SeedSequence([self.scenario_seed, 401, m])))
                        for m in range(c.machines)]
        self.period = 0
        self.peak_kw = float(physical.uniform(20, 50)) if c.random_initial_peak else c.initial_peak_kw
        self.initial_peak = self.peak_kw
        self.period_peaks = []
        self.billing_ledger, self.daily_ledger = [], []
        self.period_demand_start = 0.0
        self.window_kwh = self.energy_kwh = self.energy_cost = self.demand_liability = 0.0
        self.tardiness = self.maintenance_downtime = self.repair_downtime = 0.0
        self.inspection_downtime = self.inspection_direct_cost = 0.0
        self.maintenance_count = self.inspection_count = self.breakdowns = self.productive_minutes = 0
        self.reward_total = 0.0
        self.health_error = self.health_coverage = self.health_width = self.health_samples = 0.0
        self.events, self.trace, self.demand_windows = [], [], []
        self._seek_decision()
        return self._observation(), self._info()

    def eligible(self, machine):
        indices = np.flatnonzero((self.arrivals <= self.time) & (self.completion < 0) & ~self.active)
        return [int(j) for j in indices if machine in self.jobs[j].operations[self.operation[j]]]

    def candidates(self):
        if self.current_machine < 0 or self.terminated:
            return []
        return sorted(self.eligible(self.current_machine),
                      key=lambda j: (self.jobs[j].due, self.jobs[j].arrival, j))[:self.config.shortlist]

    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[self.wait_action] = True
        if self.current_machine >= 0 and not self.terminated:
            machine = self.machines[self.current_machine]
            mask[:len(self.candidates())] = True
            mask[self.maintenance_action] = self.config.maintenance_enabled and self.open_minutes >= machine.pm_ready
            mask[self.inspection_action] = self.inspections_enabled and self.open_minutes >= machine.inspection_ready
        return mask

    def setup_time(self, machine, job):
        if not self.config.setup_enabled or machine.product in (-1, job.product):
            return 0
        return 5 + 3 * ((job.product - machine.product) % self.config.products)

    def _operating(self):
        return self.time % 1440 < self.config.shift_minutes

    def _roll_billing(self):
        new_period = ((self.time + self.config.start_hour * 60) // 1440) // self.config.billing_days
        while self.period < new_period:
            self.period_peaks.append(self.peak_kw)
            self.billing_ledger.append(dict(period=self.period, peak_kw=self.peak_kw,
                demand_cost=self.demand_liability-self.period_demand_start))
            self.period_demand_start = self.demand_liability
            self.peak_kw = 0.0
            self.period += 1

    def _advance_closed(self, target):
        target = min(target, self.config.horizon)
        if hasattr(self, 'rollout_limit'):
            target = min(target, self.rollout_limit)
        pending = self.completion < 0
        self.tardiness += float(np.maximum(0, target - self.dues[pending]).sum()
                                - np.maximum(0, self.time - self.dues[pending]).sum())
        self.time = target
        self.window_kwh = 0.0  # Remaining closed-window minutes have zero load.
        for machine in self.machines:
            machine.continuous = 0
        self._roll_billing()

    def _seek_decision(self):
        c = self.config
        while self.time < c.horizon:
            if hasattr(self, 'rollout_limit') and self.time >= self.rollout_limit:
                self.current_machine = -1
                return
            if not self._operating():
                self._advance_closed((self.time // 1440 + 1) * 1440)
                continue
            for m, machine in enumerate(self.machines):
                if machine.mode == 'idle' and self.open_minutes >= machine.defer_until:
                    if self.open_minutes >= machine.next_control or self.eligible(m):
                        self.current_machine = m
                        return
            self._tick()
        self.terminated, self.current_machine = True, -1
        if self.daily_ledger:
            self.daily_ledger[-1].update(self.cost_components())

    def step(self, action):
        if self.terminated:
            raise RuntimeError('Reset an ended episode before stepping')
        raw = np.asarray(action)
        if raw.ndim != 0 or raw.dtype.kind not in 'iu' or not 0 <= int(raw) < self.action_space.n or not self.action_masks()[int(raw)]:
            self.invalid_actions += 1
            if self.strict_actions:
                raise ValueError('Invalid inspection-environment action')
            action = self.wait_action
        else:
            action = int(raw)
        before = self.cost_components()['full']
        c, m = self.config, self.current_machine
        machine = self.machines[m]
        if action < c.shortlist:
            j = self.candidates()[action]
            machine.job = j
            machine.processing_left = self.jobs[j].operations[self.operation[j]][m]
            setup = self.setup_time(machine, self.jobs[j])
            machine.mode = 'setup' if setup else 'process'
            machine.remaining = setup if setup else machine.processing_left
            self.active[j] = True
        elif action == self.maintenance_action:
            machine.mode, machine.remaining = 'maintenance', c.maintenance_minutes
            self.maintenance_count += 1
        elif action == self.inspection_action:
            machine.mode, machine.remaining = 'inspection', c.inspection_minutes
            self.inspection_count += 1
            self.inspection_direct_cost += c.inspection_cost
        else:
            machine.defer_until = self.open_minutes + c.wait_minutes
            machine.next_control = self.open_minutes + c.control_minutes
        if self.record:
            self.events.append(dict(time=self.time, machine=m, action=action, job=machine.job))
        self.steps += 1
        self._seek_decision()
        reward = -(self.cost_components()['full'] - before) / c.reward_scale
        self.reward_total += reward
        return self._observation(), float(reward), self.terminated, False, self._info()

    def _account_power(self, power):
        c = self.config
        self._roll_billing()
        absolute = c.start_hour * 60 + self.time
        hour = (absolute / 60) % 24
        tariff = c.peak_rate if 17 <= hour < 22 else c.offpeak_rate if hour < 6 or hour >= 22 else c.normal_rate
        self.energy_kwh += power / 60
        self.energy_cost += power / 60 * tariff
        self.window_kwh += power / 60
        average = self.window_kwh * 60 / c.demand_window
        if average > self.peak_kw:
            if c.demand_enabled:
                self.demand_liability += (average - self.peak_kw) * c.demand_rate
            self.peak_kw = average
        if (absolute + 1) % c.demand_window == 0:
            if self.record:
                self.demand_windows.append(dict(end_minute=self.time+1, period=self.period, average_kw=average))
            self.window_kwh = 0.0

    def _tick(self):
        c = self.config
        # Jobs overdue at t incur one minute through t+1, including final completion.
        self.tardiness += int(np.sum((self.dues <= self.time) & (self.completion < 0)))
        power = 0.0
        for m, machine in enumerate(self.machines):
            mode = machine.mode
            product = self.jobs[machine.job].product if machine.job >= 0 else 0
            kw = machine.power * (0.8 + 0.2 * product) if mode == 'process' else machine.power * 0.4 if mode == 'setup' else 0.8 if mode == 'idle' else 0.2
            power += kw
            if self.record:
                self.trace.append(dict(time=self.time, machine=m, mode=mode, job=machine.job, power_kw=kw))
            if mode == 'idle':
                machine.continuous = 0
                continue
            machine.remaining -= 1
            if mode in ('process', 'setup'):
                machine.continuous += 1
                decrement = c.wear_strength * (c.wear_per_minute * kw / 20 + c.continuous_wear * machine.continuous / 120)
                machine.health = max(0.0, machine.health - decrement * self.true_rates[m] * self.wear_shocks[self.time, m])
                slope = c.failure_health if c.load_dependent_failure else 0.0
                base = c.failure_base if c.load_dependent_failure else c.failure_base + 0.25*c.failure_health
                hazard = np.clip(base + slope*(1-machine.health), 0, 1)
                failed = bool(machine.remaining > 0 and self.failure_shocks[self.time, m] < hazard)
                self.beliefs[m].predict(decrement, failed if machine.remaining > 0 else None, base, slope)
                if mode == 'process':
                    self.productive_minutes += 1
                if failed:
                    machine.resume_mode, machine.resume_left = mode, machine.remaining
                    machine.mode, machine.remaining = 'repair', c.repair_minutes
                    self.breakdowns += 1
                    if self.record:
                        self.events.append(dict(time=self.time+1, machine=m, event='breakdown'))
                    continue
            elif mode == 'maintenance':
                self.maintenance_downtime += 1
            elif mode == 'inspection':
                self.inspection_downtime += 1
            elif mode == 'repair':
                self.repair_downtime += 1
            if machine.remaining == 0:
                if mode == 'setup':
                    machine.mode, machine.remaining = 'process', machine.processing_left
                elif mode == 'process':
                    j = machine.job
                    machine.product = self.jobs[j].product
                    self.operation[j] += 1
                    self.active[j] = False
                    if self.operation[j] == len(self.jobs[j].operations):
                        self.completion[j] = self.time + 1
                    machine.mode, machine.job = 'idle', -1
                elif mode == 'maintenance':
                    machine.health, machine.continuous, machine.mode = 1.0, 0, 'idle'
                    machine.last_maintenance = self.time + 1
                    machine.pm_ready = self.open_minutes + 1 + c.maintenance_cooldown
                    self.beliefs[m].restore(1.0)
                elif mode == 'repair':
                    machine.health, machine.continuous = 0.8, 0
                    machine.mode, machine.remaining = machine.resume_mode, machine.resume_left
                    self.beliefs[m].restore(0.8)
                elif mode == 'inspection':
                    measurement = float(np.clip(machine.health + c.inspection_noise*self.sensor_shocks[self.time, m], 0, 1))
                    self.beliefs[m].observe(measurement, c.inspection_noise)
                    machine.last_measurement = self.time + 1
                    machine.inspection_ready = self.open_minutes + 1 + c.inspection_minutes
                    machine.mode = 'idle'
                    if self.record:
                        self.events.append(dict(time=self.time+1, machine=m, event='measurement', value=measurement))
        self._account_power(power)
        self.time += 1
        self.open_minutes += 1
        if self.time % 1440 == c.shift_minutes:
            released = self.arrivals <= self.time
            self.daily_ledger.append(dict(day=self.time//1440, time=self.time,
                released=int(released.sum()),completed=int((self.completion>=0).sum()),
                backlog=int((released&(self.completion<0)).sum()),period=self.period,
                peak_kw=self.peak_kw,energy_cost=self.energy_cost,demand_cost=self.demand_liability,
                total_tardiness_min=self.tardiness,**self.cost_components()))
        if self.open_minutes % 15 == 0:
            for machine, belief in zip(self.machines, self.beliefs):
                summary = belief.summary()
                self.health_error += abs(machine.health-summary['mean'])
                self.health_coverage += summary['q05'] <= machine.health <= summary['q95']
                self.health_width += summary['q95']-summary['q05']
                self.health_samples += 1

    def cost_components(self):
        c = self.config
        lateness = c.tardiness_weight * self.tardiness / c.tardiness_scale
        terminal = c.unfinished_penalty * int(np.sum(self.completion < 0)) if self.terminated else 0.0
        energy = c.energy_weight*self.energy_cost/c.energy_scale
        demand = c.demand_weight*self.demand_liability/c.demand_scale
        downtime = c.maintenance_weight*(self.maintenance_downtime+self.repair_downtime+self.inspection_downtime)/c.downtime_scale
        inspection = self.inspection_direct_cost / c.energy_scale
        return dict(lateness=lateness, terminal=terminal, energy=energy, demand=demand,
                    downtime=downtime, inspection=inspection,
                    full=lateness+terminal+energy+demand+downtime+inspection)

    def policy_state(self):
        c = self.config
        machines = []
        for m, (machine, belief) in enumerate(zip(self.machines, self.beliefs)):
            condition = belief.summary()
            if self.exact_health:
                condition = dict(mean=machine.health, std=0.0, q05=machine.health,
                                 q95=machine.health, median=machine.health)
            machines.append(dict(**condition, mode=machine.mode, remaining=machine.remaining,
                power=machine.power, continuous=machine.continuous, product=machine.product,
                maintenance_age=self.time-machine.last_maintenance,
                measurement_age=self.time-machine.last_measurement if machine.last_measurement >= 0 else -1,
                measured=machine.last_measurement >= 0, defer=max(0, machine.defer_until-self.open_minutes),
                pm_cooldown=max(0, machine.pm_ready-self.open_minutes), eligible=len(self.eligible(m))))
        candidates = []
        if self.current_machine >= 0:
            machine = self.machines[self.current_machine]
            for j in self.candidates():
                job, op = self.jobs[j], self.operation[j]
                work = sum(min(o.values()) for o in job.operations[op:])
                candidates.append(dict(job=j, duration=job.operations[op][self.current_machine],
                    slack=job.due-self.time-work, setup=self.setup_time(machine, job),
                    operations_left=len(job.operations)-op, power=machine.power*(0.8+0.2*job.product),
                    age=self.time-job.arrival, product=job.product, due=job.due))
        waiting = int(np.sum((self.arrivals <= self.time)&(self.completion < 0)&~self.active))
        late = int(np.sum((self.dues < self.time)&(self.completion < 0)))
        return dict(time=self.time, open_minutes=self.open_minutes, machine=self.current_machine,
                    machines=machines, candidates=candidates, waiting=waiting, late=late,
                    peak_kw=self.peak_kw, period=self.period,
                    action_mask=self.action_masks().copy())

    def _observation(self):
        c, state = self.config, self.policy_state()
        absolute = self.time+c.start_hour*60
        clock = 2*np.pi*(absolute % 1440)/1440
        elapsed = self.time / c.horizon
        obs = [elapsed, np.sin(clock), np.cos(clock), c.tariff(self.time)/max(1,c.peak_rate),
               self.peak_kw/300, self.window_kwh*60/c.demand_window/300,
               (absolute % c.demand_window)/c.demand_window, state['waiting']/c.jobs,
               state['late']/c.jobs, max(0,c.shift_minutes-self.time%1440)/c.shift_minutes,
               1-elapsed, ((absolute//1440)%c.billing_days)/c.billing_days,
               max(0,math.ceil(c.days/c.billing_days)-self.period)/max(1,math.ceil(c.days/c.billing_days)),
               max(0,self.current_machine)/c.machines, float(self.inspections_enabled), c.inspection_cost/1000,
               c.inspection_noise]
        for slot in range(c.shortlist):
            if slot >= len(state['candidates']):
                obs.extend([0.]*(len(JOB_FEATURES)+c.products))
                continue
            job = state['candidates'][slot]
            obs.extend([1.,job['duration']/60,job['slack']/c.shift_minutes,job['setup']/30,
                        job['operations_left']/c.max_operations,job['power']/50,job['age']/1440,
                        *[float(job['product']==p) for p in range(c.products)]])
        for m in range(c.machine_slots):
            if m >= c.machines:
                obs.extend([0.]*(len(MACHINE_FEATURES)+c.products))
                continue
            machine = state['machines'][m]
            obs.extend([1.,float(m==self.current_machine),machine['mean'],machine['std'],machine['q05'],machine['q95'],
                        float(machine['mode']!='idle'),machine['remaining']/c.shift_minutes,machine['eligible']/c.jobs,
                        machine['power']/50,machine['continuous']/c.shift_minutes,machine['maintenance_age']/1440,
                        max(0,machine['measurement_age'])/1440,float(machine['measured']),
                        machine['defer']/c.wait_minutes,machine['pm_cooldown']/c.maintenance_cooldown,
                        *[float(machine['product']==p) for p in range(c.products)]])
        return np.asarray(obs,dtype=np.float32)

    def _info(self):
        metrics = self.metrics()
        # Truth-based inference diagnostics are evaluator-only, not policy input.
        metrics = {key:value for key,value in metrics.items() if not key.startswith(('belief_','ledger_'))}
        return dict(time=self.time, machine=self.current_machine, action_mask=self.action_masks(),
                    metrics=metrics, cost_components=self.cost_components())

    def metrics(self):
        released = self.arrivals <= self.time
        done = self.completion >= 0
        lateness = np.maximum(0,np.where(done,self.completion,self.time)-self.dues)
        number = max(1,int(released.sum()))
        return dict(full_objective=self.cost_components()['full'], completed_jobs=int(done.sum()),
                    released_jobs=int(released.sum()), unfinished_jobs=int((released&~done).sum()),
                    completion_rate=float(done.sum()/number), on_time_rate=float(np.sum(done&(lateness==0))/number),
                    total_tardiness_min=float(self.tardiness), average_tardiness_min=float(lateness[released].sum()/number),
                    p95_tardiness_min=float(np.quantile(lateness[released],.95)) if released.any() else 0.,
                    breakdowns=self.breakdowns, maintenance_count=self.maintenance_count,
                    inspection_count=self.inspection_count, maintenance_downtime_min=self.maintenance_downtime,
                    repair_downtime_min=self.repair_downtime, inspection_downtime_min=self.inspection_downtime,
                    inspection_direct_cost=self.inspection_direct_cost, energy_kwh=self.energy_kwh,
                    energy_cost=self.energy_cost, demand_cost=self.demand_liability, billing_peak_kw=self.peak_kw,
                    invalid_actions=self.invalid_actions, decisions=self.steps, calendar_minutes=self.time,
                    belief_health_mae=self.health_error/max(1,self.health_samples),
                    belief_coverage90=self.health_coverage/max(1,self.health_samples),
                    belief_interval_width=self.health_width/max(1,self.health_samples),
                    belief_fallbacks=sum(b.fallbacks for b in self.beliefs),
                    ledger_daily=self.daily_ledger.copy(),
                    ledger_billing=self.billing_ledger + [dict(period=self.period,peak_kw=self.peak_kw,
                        demand_cost=self.demand_liability-self.period_demand_start)])

    def snapshot(self):
        # Shock tables are immutable; all mutable state and filter RNGs are copied.
        shared = {id(x):x for x in (self.failure_shocks,self.wear_shocks,self.sensor_shocks)}
        return copy.deepcopy(self, shared)
