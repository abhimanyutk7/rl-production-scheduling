from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from functools import lru_cache


TAILLARD_TIME_SEEDS = (840612802, 1314640371, 1227221349, 342269428, 1603221416, 1357584978,
                       44531661, 302545136, 1153780144, 73896786)


def taillard_matrix(jobs, machines, time_seed, machine_seed):
    """Taillard generator, reproduced from OR-Library jobshop2; zero-based machine IDs."""
    def uniform(seed, low, high):
        seed = (16807 * seed) % 2147483647
        return seed, low + int(seed / 2147483647 * (high - low + 1))
    times, routes = [], []
    for _ in range(jobs):
        row = []
        for _ in range(machines):
            time_seed, duration = uniform(time_seed, 1, 99)
            row.append(duration)
        times.append(row)
    for _ in range(jobs):
        row = list(range(machines))
        for i in range(machines):
            machine_seed, other = uniform(machine_seed, i, machines - 1)
            row[i], row[other] = row[other], row[i]
        routes.append(row)
    return times, routes


@lru_cache(maxsize=10)
def duration_pool(index):
    return np.array(taillard_matrix(15, 15, TAILLARD_TIME_SEEDS[index], 398197754)[0]).ravel()


@dataclass
class Job:
    arrival: int
    due: int
    product: int
    operations: list[dict[int, int]]


def generate_jobs(config, rng, scenario_seed=None):
    jobs = []
    # A finite stream of Poisson arrivals; the first order is available at shift start.
    interarrival = rng.exponential(config.arrival_minutes / max(1, config.jobs - 1), config.jobs - 1)
    arrivals = np.r_[0, np.ceil(np.cumsum(interarrival)).astype(int)]
    source_indices = range(8, 10) if scenario_seed is not None and scenario_seed >= 20000 else range(6, 8) if scenario_seed is not None and scenario_seed >= 10000 else range(6)
    durations = duration_pool(int(rng.choice(list(source_indices))))
    for arrival in arrivals:
        ops = []
        for _ in range(int(rng.integers(1, config.max_operations + 1))):
            machines = rng.choice(config.machines, size=min(2, config.machines), replace=False)
            duration = int(rng.choice(durations))
            ops.append({int(m): max(1, int(round(duration * config.benchmark_time_scale / (1 + 0.1 * (m % 5))))) for m in machines})
        nominal = sum(min(op.values()) for op in ops)
        due = int(arrival + nominal * rng.uniform(1.7, 3.0) + 30)
        jobs.append(Job(int(arrival), due, int(rng.integers(config.products)), ops))
    return jobs


def validate_jobs(jobs, config):
    if len(jobs) != config.jobs:
        raise ValueError("Instance job count differs from config.jobs")
    for job in jobs:
        if not 0 <= job.arrival < config.max_minutes or job.due < job.arrival:
            raise ValueError("Invalid arrival or due date")
        if not 0 <= job.product < config.products or not 1 <= len(job.operations) <= config.max_operations:
            raise ValueError("Invalid product or operation count")
        for op in job.operations:
            if not op or any(not 0 <= m < config.machines or not isinstance(d, int) or d < 1 for m, d in op.items()):
                raise ValueError("Each operation requires valid machines and positive integer durations")
    return jobs


def load_instance(path, config):
    data = json.loads(Path(path).read_text())
    jobs = [Job(j["arrival"], j["due"], j["product"], [{int(m): d for m, d in op.items()} for op in j["operations"]]) for j in data["jobs"]]
    return validate_jobs(jobs, config)


def import_benchmark(path, format, due_factor=2.5):
    """Import standard pair-list JSSP or Brandimarte FJSP text without invented measurements."""
    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    header = [int(x) for x in lines[0].split()]
    n_jobs, n_machines = header[:2]
    if n_jobs < 1 or n_machines < 1 or len(lines) != n_jobs + 1:
        raise ValueError("Expected header and exactly one line per job")
    jobs = []
    for i, line in enumerate(lines[1:]):
        values = [int(x) for x in line.split()]
        operations = []
        if format == "jssp":
            if len(values) != 2 * n_machines:
                raise ValueError("JSSP rows must contain machine/duration pairs (zero-based machine IDs)")
            operations = [{values[k]: values[k + 1]} for k in range(0, len(values), 2)]
        else:
            count, cursor = values[0], 1
            for _ in range(count):
                if cursor >= len(values):
                    raise ValueError("Truncated FJSP operation")
                alternatives = values[cursor]
                cursor += 1
                if alternatives < 1 or cursor + 2 * alternatives > len(values):
                    raise ValueError("Invalid FJSP alternatives")
                op = {}
                for _ in range(alternatives):
                    machine, duration = values[cursor:cursor + 2]
                    if machine - 1 in op:
                        raise ValueError("Duplicate alternative machine")
                    op[machine - 1] = duration
                    cursor += 2
                operations.append(op)
            if cursor != len(values):
                raise ValueError("Extra FJSP row values")
        jobs.append(Job(0, int(due_factor * sum(min(op.values()) for op in operations)), i % 3, operations))
    from .config import ShopConfig
    max_ops = max(map(lambda j: len(j.operations), jobs))
    horizon = max(1440, 2 * sum(sum(max(op.values()) for op in j.operations) for j in jobs))
    config = ShopConfig(machines=n_machines, machine_slots=max(10, n_machines), jobs=n_jobs, max_operations=max_ops, arrival_minutes=0, shift_minutes=horizon, max_minutes=horizon,
                        setup_enabled=False, maintenance_enabled=False, wear_strength=0, failure_base=0, failure_health=0)
    validate_jobs(jobs, config)
    return {"provenance": str(Path(path).resolve()), "format": format, "augmentation": "Synthetic due dates, products, power and tariff; processing durations are from the input.",
            "jobs": [{"arrival": j.arrival, "due": j.due, "product": j.product, "operations": j.operations} for j in jobs]}, config
