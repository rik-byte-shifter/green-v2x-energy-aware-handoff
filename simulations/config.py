from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Simulation Configuration Parameters"""

    area_size: int = 2000
    num_base_stations: int = 9
    bs_coverage_radius: float = 300.0

    num_vehicles: int = 20
    vehicle_speed_min: float = 10.0
    vehicle_speed_max: float = 30.0

    duration: int = 1000
    time_step: float = 1.0

    tx_power_default: float = 0.1
    data_rate: float = 1e6
    packet_size: int = 1000

    seed: int = 42

    # Extra energy per handoff (signaling / re-association), joules
    handoff_energy_joules: float = 0.08

    # Minimum time between handoffs for one vehicle (seconds), reduces ping-pong
    handoff_cooldown_s: float = 4.0

    def __post_init__(self):
        assert self.num_vehicles > 0, "Need at least 1 vehicle"
        assert self.duration > 0, "Duration must be positive"
        assert self.time_step > 0, "Time step must be positive"


DEFAULT_CONFIG = SimulationConfig()
