from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ShopConfig:
    machines: int = 5
    jobs: int = 50
    max_operations: int = 3
    products: int = 3
    shortlist: int = 10
    machine_slots: int = 10
    benchmark_time_scale: float = 0.5
    shift_minutes: int = 480
    arrival_minutes: int = 240
    max_minutes: int = 1440
    start_hour: int = 8
    demand_window: int = 15
    initial_peak_kw: float = 30.0
    random_initial_peak: bool = True
    demand_rate: float = 100.0
    offpeak_rate: float = 4.0
    normal_rate: float = 7.0
    peak_rate: float = 12.0
    wear_strength: float = 1.0
    wear_per_minute: float = 0.0015
    continuous_wear: float = 0.0002
    health_visible: bool = True
    load_dependent_failure: bool = True
    failure_base: float = 0.0003
    failure_health: float = 0.008
    maintenance_minutes: int = 30
    repair_minutes: int = 90
    maintenance_enabled: bool = True
    setup_enabled: bool = True
    demand_enabled: bool = True
    tardiness_weight: float = 1.0
    energy_weight: float = 1.0
    demand_weight: float = 1.0
    maintenance_weight: float = 0.2
    reward_scale: float = 1.0
    tardiness_scale: float = 480.0
    energy_scale: float = 1000.0
    demand_scale: float = 3000.0
    downtime_scale: float = 60.0
    unfinished_penalty: float = 2.0

    def __post_init__(self):
        for name in ("machines", "jobs", "max_operations", "products", "shortlist", "machine_slots", "shift_minutes", "max_minutes", "demand_window", "maintenance_minutes", "repair_minutes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.arrival_minutes) is not int or not 0 <= self.arrival_minutes < self.max_minutes:
            raise ValueError("arrival_minutes must be an integer below max_minutes")
        if type(self.start_hour) is not int or not 0 <= self.start_hour < 24:
            raise ValueError("start_hour must be 0..23")
        if self.shift_minutes > self.max_minutes:
            raise ValueError("shift_minutes must not exceed max_minutes")
        if self.machines > self.machine_slots:
            raise ValueError("machines must not exceed machine_slots")
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is bool and type(value) is not bool:
                raise ValueError(f"{field.name} must be a boolean")
            if field.type is float and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{field.name} must be finite and non-negative")
        if any(getattr(self, n) <= 0 for n in ("reward_scale", "unfinished_penalty", "tardiness_scale", "energy_scale", "demand_scale", "downtime_scale")):
            raise ValueError("Cost scales and unfinished penalty must be positive")
        if self.failure_base + self.failure_health > 1:
            raise ValueError("failure probabilities must sum to at most one")
        if self.benchmark_time_scale <= 0:
            raise ValueError("benchmark_time_scale must be positive")

    @classmethod
    def load(cls, path=None):
        return cls(**json.loads(Path(path).read_text())) if path else cls()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    def tariff(self, minute):
        hour = (self.start_hour + minute / 60) % 24
        return self.peak_rate if 17 <= hour < 22 else self.offpeak_rate if hour < 6 or hour >= 22 else self.normal_rate
