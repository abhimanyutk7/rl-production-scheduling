import time
from ortools.sat.python import cp_model


def dispatch(env, rule="spt"):
    candidates = env.candidates()
    machine = env.machines[env.current_machine]
    if rule == "health" and machine.health < 0.55 and env.action_masks()[env.maintenance_action]:
        return env.maintenance_action
    if rule == "edd":
        return 0
    key = (lambda j: (env.jobs[j].arrival, j)) if rule == "fcfs" else (
        lambda j: (env.jobs[j].operations[env.operation[j]][env.current_machine] + env.setup_time(machine, env.jobs[j]), j))
    return candidates.index(min(candidates, key=key))


class CPSATPolicy:
    """Time-limited schedule baseline; no claim to solve stochastic wear or the full bill."""

    def __init__(self, rolling=True, time_limit=0.1):
        self.rolling, self.time_limit = rolling, time_limit
        self.plan = {}
        self.signature = None
        self.statuses = []
        self.solve_seconds = 0.0

    def __call__(self, env):
        signature = (tuple(j for j, job in enumerate(env.jobs) if job.arrival <= env.time), env.breakdowns, env.maintenance_count)
        if self.signature is None or self.rolling and signature != self.signature:
            self._solve(env, visible_only=self.rolling)
            self.signature = signature
        candidates = env.candidates()
        return min(range(len(candidates)), key=lambda a: self.plan.get((candidates[a], int(env.operation[candidates[a]]), env.current_machine), 10**9))

    def _solve(self, env, visible_only):
        c = env.config
        model = cp_model.CpModel()
        horizon = c.max_minutes + sum(sum(max(op.values()) + 15 for op in job.operations) for job in env.jobs)
        intervals = [[] for _ in env.machines]
        availability, initial_product = [], []
        for m, machine in enumerate(env.machines):
            duration = machine.remaining
            if machine.mode == "setup":
                duration += machine.processing_left
            elif machine.mode == "repair":
                duration += machine.resume_left + (machine.processing_left if machine.resume_mode == "setup" else 0)
            availability.append(env.time + duration)
            initial_product.append(env.jobs[machine.job].product if machine.job >= 0 else machine.product)
        starts, ends, ready, alternatives, on_machine = {}, {}, {}, {}, [[] for _ in env.machines]
        tardiness = []
        hint_available, hint_product = availability.copy(), initial_product.copy()
        hint_start, hint_end, hint_machine = {}, {}, {}
        for j, job in enumerate(env.jobs):
            if env.completion[j] >= 0 or visible_only and job.arrival > env.time:
                continue
            first = int(env.operation[j]) + int(env.active[j])
            previous = next((availability[m] for m, machine in enumerate(env.machines) if machine.job == j), env.time)
            previous_hint = previous
            for k in range(first, len(job.operations)):
                start = model.new_int_var(max(env.time, job.arrival), horizon, f"s{j}_{k}")
                end = model.new_int_var(0, horizon, f"e{j}_{k}")
                starts[j, k], ends[j, k] = start, end
                ready[j, k] = previous
                model.add(start >= previous)
                present = []
                for m, duration in job.operations[k].items():
                    flag = model.new_bool_var(f"a{j}_{k}_{m}")
                    interval = model.new_optional_interval_var(start, duration, end, flag, f"i{j}_{k}_{m}")
                    model.add(start >= availability[m]).only_enforce_if(flag)
                    intervals[m].append(interval)
                    alternatives[j, k, m] = flag
                    on_machine[m].append((j, k, flag))
                    present.append(flag)
                model.add_exactly_one(present)
                def earliest_completion(m):
                    product = job.product
                    setup = 0 if not c.setup_enabled or hint_product[m] in (-1, product) else 5 + 3 * ((product - hint_product[m]) % c.products)
                    start_hint = max(env.time, job.arrival, previous_hint, hint_available[m]) + setup
                    return start_hint + job.operations[k][m], start_hint, m
                end_hint, start_hint, chosen = min(earliest_completion(m) for m in job.operations[k])
                hint_start[j, k], hint_end[j, k], hint_machine[j, k] = start_hint, end_hint, chosen
                hint_available[chosen], hint_product[chosen] = end_hint, job.product
                model.add_hint(start, start_hint)
                model.add_hint(end, end_hint)
                for m in job.operations[k]:
                    model.add_hint(alternatives[j, k, m], int(m == chosen))
                previous_hint = end_hint
                previous = end
            late = model.new_int_var(0, horizon, f"late{j}")
            model.add(late >= previous - job.due)
            model.add_hint(late, max(0, previous_hint - job.due))
            tardiness.append(late)
        for m, machine_tasks in enumerate(on_machine):
            model.add_no_overlap(intervals[m])
            if not c.setup_enabled or not machine_tasks:
                continue
            empty = model.new_bool_var(f"empty{m}")
            arcs = [(0, 0, empty)]
            order = sorted([i for i, (j, k, _) in enumerate(machine_tasks, 1) if hint_machine[j, k] == m],
                           key=lambda i: hint_start[machine_tasks[i - 1][:2]])
            model.add_hint(empty, int(not order))
            for i, (j, k, flag) in enumerate(machine_tasks, 1):
                arcs.append((i, i, flag.Not()))
                first_flag = model.new_bool_var(f"first{m}_{i}")
                last_flag = model.new_bool_var(f"last{m}_{i}")
                arcs.extend([(0, i, first_flag), (i, 0, last_flag)])
                model.add_hint(first_flag, int(bool(order) and order[0] == i))
                model.add_hint(last_flag, int(bool(order) and order[-1] == i))
                product = env.jobs[j].product
                setup = 0 if initial_product[m] in (-1, product) else 5 + 3 * ((product - initial_product[m]) % c.products)
                model.add(starts[j, k] >= availability[m] + setup).only_enforce_if(first_flag)
                model.add(starts[j, k] >= ready[j, k] + setup).only_enforce_if(first_flag)
                model.add(starts[j, k] >= env.jobs[j].arrival + setup).only_enforce_if(first_flag)
                for q, (jj, kk, _) in enumerate(machine_tasks, 1):
                    if q == i:
                        continue
                    follows = model.new_bool_var(f"next{m}_{i}_{q}")
                    arcs.append((i, q, follows))
                    model.add_hint(follows, int(i in order and order.index(i) + 1 < len(order) and order[order.index(i) + 1] == q))
                    next_product = env.jobs[jj].product
                    setup = 0 if next_product == product else 5 + 3 * ((next_product - product) % c.products)
                    model.add(starts[jj, kk] >= ends[j, k] + setup).only_enforce_if(follows)
                    model.add(starts[jj, kk] >= ready[jj, kk] + setup).only_enforce_if(follows)
                    model.add(starts[jj, kk] >= env.jobs[jj].arrival + setup).only_enforce_if(follows)
            model.add_circuit(arcs)
        if not starts:
            return
        makespan = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan, list(ends.values()))
        model.add_hint(makespan, max(hint_end.values()))
        model.minimize(10 * sum(tardiness) + makespan)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        # Preserve a complete feasible hint under short limits instead of exhausting them in presolve.
        solver.parameters.cp_model_presolve = False
        started = time.perf_counter()
        status = solver.solve(model)
        self.solve_seconds += time.perf_counter() - started
        self.statuses.append(solver.status_name(status))
        self.plan = {}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self.plan = {(j, k, m): solver.value(starts[j, k]) for (j, k, m), flag in alternatives.items() if solver.value(flag)}
        # shortcut: a time-limited solver may have no incumbent; EDD is the explicit fallback.
