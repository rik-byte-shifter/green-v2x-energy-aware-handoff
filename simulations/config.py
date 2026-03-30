from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from src.models.weather import WeatherProfile, WeatherType, WEATHER_PROFILES, get_weather_profile


@dataclass
class SimulationConfig:
    """Simulation Configuration Parameters"""

    area_size: int = 3000
    num_base_stations: int = 9
    bs_coverage_radius: float = 250.0

    num_vehicles: int = 20
    vehicle_speed_min: float = 10.0
    vehicle_speed_max: float = 30.0

    # "highway": straight road with fixed lanes (direction along +x by default)
    # "area": legacy random walk with random direction and reflections
    movement_mode: Literal["highway", "area"] = "highway"
    highway_num_lanes: int = 4
    highway_lane_width_m: float = 4.0
    highway_direction_rad: float = 0.0

    # Rare lateral moves: adjacent lane only, mean rate ~ prob_per_s * dt per vehicle
    highway_lane_switch_prob_per_s: float = 0.0008
    highway_lane_switch_cooldown_s: float = 10.0

    # Per-lane speed ranges (m/s). If None, lane 0 is slowest → lane n-1 fastest.
    highway_lane_speed_min: Optional[List[float]] = None
    highway_lane_speed_max: Optional[List[float]] = None

    duration: int = 1000
    time_step: float = 1.0

    tx_power_default: float = 0.1
    data_rate: float = 1e6
    packet_size: int = 1000

    seed: int = 42

    # Extra energy per handoff (signaling / re-association), joules
    handoff_energy_joules: float = 0.08

    # During handoff the radio may stay on; energy += total TX-chain power * handoff_delay (J)
    handoff_delay_s: float = 0.05

    # Minimum time between handoffs for one vehicle (seconds), reduces ping-pong
    handoff_cooldown_s: float = 4.0
    # Ping-pong detection window (seconds): A->B then B->A within window.
    ping_pong_window_s: float = 10.0

    # Energy-aware: min relative EPB gain to handoff (0.25 = 25%); TTT reduces ping-pong
    energy_aware_min_energy_saving: float = 0.25
    energy_aware_time_to_trigger_s: float = 2.0
    # QoS: minimum Shannon-style achievable rate (bps) to prefer a BS; 0 = no floor
    energy_aware_min_data_rate_bps: float = 5e6

    # SNR (dB) must exceed this for a non-outage link; at or below → data rate 0
    snr_outage_threshold_db: float = 0.0

    # Applied to BSConfig: slow-fading / shadowing on RX power (dB). 0 = off.
    shadowing_std_db: float = 8.0
    # Reliability target for percentile-margin TX provisioning under shadowing.
    # Example: 0.95 -> z ~= 1.645, margin = z * sigma_shadowing_db.
    shadowing_reliability: float = 0.95
    # Target RX threshold used by the TX power requirement function.
    target_rx_power_dbm: float = -90.0

    # Weather affects propagation (path loss exponent, shadowing, and extra
    # precipitation/scattering attenuation).
    weather_profile: WeatherType = "clear"

    # Lateral offset around lane center (m, highway mode only). 0 = exact lane center.
    highway_lateral_noise_std_m: float = 0.5

    # Grid carbon intensity for reporting CO2 from energy (kg CO2 per kWh).
    carbon_intensity_kg_per_kwh: float = 0.5

    # Mean Gregorian year; used to annualize per-vehicle CO2 from simulation duration.
    seconds_per_year: float = 365.25 * 24 * 3600

    # If True, RSSI baseline uses fixed tx_power_default + data_rate for energy (classic static TX);
    # if False, RSSI uses the same Shannon link model as reporting (fair PHY comparison).
    rssi_energy_use_fixed_tx: bool = False

    def highway_lane_centers_y(self) -> List[float]:
        """Y positions of lane centers, highway block centered in the map."""
        n = self.highway_num_lanes
        w = self.highway_lane_width_m
        bottom = (self.area_size - n * w) / 2.0
        return [bottom + (k + 0.5) * w for k in range(n)]

    def highway_lane_speed_bounds(self, lane_index: int) -> Tuple[float, float]:
        """Min/max speed (m/s) for lane_index. Lower indices ≈ slower lane by default."""
        n = self.highway_num_lanes
        if not (0 <= lane_index < n):
            raise IndexError("lane_index out of range")
        if self.highway_lane_speed_min is not None:
            assert self.highway_lane_speed_max is not None
            return (
                self.highway_lane_speed_min[lane_index],
                self.highway_lane_speed_max[lane_index],
            )
        t = lane_index / max(1, n - 1)
        lo = self.vehicle_speed_min * (0.82 + 0.18 * t)
        hi = self.vehicle_speed_max * (0.82 + 0.18 * t)
        return lo, hi

    def __post_init__(self):
        assert self.num_vehicles > 0, "Need at least 1 vehicle"
        assert self.duration > 0, "Duration must be positive"
        assert self.time_step > 0, "Time step must be positive"
        if self.movement_mode == "highway":
            assert self.highway_num_lanes >= 1
            assert self.highway_lane_width_m > 0
            assert (
                self.highway_num_lanes * self.highway_lane_width_m <= self.area_size
            ), "Highway lanes must fit inside area_size"
            assert self.highway_lane_switch_prob_per_s >= 0.0
            assert self.highway_lane_switch_cooldown_s >= 0.0
            if self.highway_lane_speed_min is not None:
                assert self.highway_lane_speed_max is not None
                assert len(self.highway_lane_speed_min) == self.highway_num_lanes
                assert len(self.highway_lane_speed_max) == self.highway_num_lanes
                for lo, hi in zip(
                    self.highway_lane_speed_min, self.highway_lane_speed_max
                ):
                    assert lo > 0 and hi >= lo
        assert self.shadowing_std_db >= 0.0
        assert 0.0 < self.shadowing_reliability < 1.0
        if self.weather_profile not in WEATHER_PROFILES:
            # Allow unknown strings from external callers/CLI; default to clear.
            self.weather_profile = "clear"
        assert self.highway_lateral_noise_std_m >= 0.0
        assert self.carbon_intensity_kg_per_kwh >= 0.0
        assert self.seconds_per_year > 0.0
        assert 0.0 < self.energy_aware_min_energy_saving < 1.0
        assert self.energy_aware_time_to_trigger_s >= 0.0
        assert self.energy_aware_min_data_rate_bps >= 0.0
        assert self.ping_pong_window_s >= 0.0

    @staticmethod
    def sparse_demonstration_scenario() -> "SimulationConfig":
        """
        High shadowing + sparse coverage feel with PHY-fair baselines:
        RSSI, energy-aware, and naive-nearest all use the same adaptive link PHY
        assumptions for energy reporting. A fixed-TX variant is available as a
        sensitivity experiment.
        """
        return SimulationConfig(
            num_vehicles=20,
            num_base_stations=9,
            area_size=3000,
            bs_coverage_radius=250,
            duration=800,
            movement_mode="highway",
            highway_num_lanes=4,
            shadowing_std_db=15.0,
            highway_lateral_noise_std_m=0.5,
            handoff_cooldown_s=8.0,
            energy_aware_min_energy_saving=0.32,
            energy_aware_time_to_trigger_s=4.5,
            energy_aware_min_data_rate_bps=1e6,
            rssi_energy_use_fixed_tx=False,
            tx_power_default=0.12,
            shadowing_reliability=0.95,
            target_rx_power_dbm=-90.0,
            weather_profile="heavy_rain",
        )

    @staticmethod
    def sparse_demonstration_scenario_fixed_rssi_tx_sensitivity() -> "SimulationConfig":
        """
        Sensitivity experiment: enable `rssi_energy_use_fixed_tx`.

        The simulator keeps the fair adaptive baseline for primary metrics/plots,
        and exports fixed-TX energy as extra fields under the RSSI results.
        """
        cfg = SimulationConfig.sparse_demonstration_scenario()
        cfg.rssi_energy_use_fixed_tx = True
        return cfg

    def get_weather(self) -> WeatherProfile:
        """Resolved weather profile configuration for the simulator."""

        return get_weather_profile(self.weather_profile)


DEFAULT_CONFIG = SimulationConfig()
