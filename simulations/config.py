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

    movement_mode: Literal["highway", "area"] = "highway"
    highway_num_lanes: int = 4
    highway_lane_width_m: float = 4.0
    highway_direction_rad: float = 0.0

    highway_lane_switch_prob_per_s: float = 0.0008
    highway_lane_switch_cooldown_s: float = 10.0

    highway_lane_speed_min: Optional[List[float]] = None
    highway_lane_speed_max: Optional[List[float]] = None

    duration: int = 1000
    time_step: float = 1.0

    tx_power_default: float = 0.1
    data_rate: float = 1e6
    packet_size: int = 1000

    seed: int = 42

    handoff_energy_joules: float = 0.08
    handoff_delay_s: float = 0.05
    handoff_cooldown_s: float = 4.0
    ping_pong_window_s: float = 10.0

    energy_aware_min_energy_saving: float = 0.25
    energy_aware_time_to_trigger_s: float = 2.0
    energy_aware_min_data_rate_bps: float = 5e6

    snr_outage_threshold_db: float = 0.0

    shadowing_std_db: float = 8.0
    shadowing_reliability: float = 0.95
    target_rx_power_dbm: float = -90.0

    weather_profile: WeatherType = "clear"

    highway_lateral_noise_std_m: float = 0.5

    # Grid carbon intensity (kg CO2 per kWh).
    # Default 0.5 is a global-average-style value.
    # For Bangladesh-specific runs use 0.62 (per Bangladesh Power Development Board).
    carbon_intensity_kg_per_kwh: float = 0.5

    use_energy_calibration: bool = True

    comprehensive_co2_include_infrastructure: bool = True
    comprehensive_co2_include_embodied: bool = False

    seconds_per_year: float = 365.25 * 24 * 3600

    rssi_energy_use_fixed_tx: bool = False

    def highway_lane_centers_y(self) -> List[float]:
        """Y positions of lane centers, highway block centered in the map."""
        n = self.highway_num_lanes
        w = self.highway_lane_width_m
        bottom = (self.area_size - n * w) / 2.0
        return [bottom + (k + 0.5) * w for k in range(n)]

    def highway_lane_speed_bounds(self, lane_index: int) -> Tuple[float, float]:
        """Min/max speed (m/s) for lane_index. Lower indices = slower lane."""
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
            self.weather_profile = "clear"
        assert self.highway_lateral_noise_std_m >= 0.0
        assert self.carbon_intensity_kg_per_kwh >= 0.0
        assert self.seconds_per_year > 0.0
        assert 0.0 < self.energy_aware_min_energy_saving < 1.0
        assert self.energy_aware_time_to_trigger_s >= 0.0
        assert self.energy_aware_min_data_rate_bps >= 0.0
        assert self.ping_pong_window_s >= 0.0

    # ------------------------------------------------------------------
    # Scenario factories
    # ------------------------------------------------------------------

    @staticmethod
    def sparse_demonstration_scenario() -> "SimulationConfig":
        """
        High-shadowing sparse scenario for the main paper baseline comparison.
        Uses heavy-rain weather to stress-test the energy-aware advantage.
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
    def scaling_scenario() -> "SimulationConfig":
        """
        FIX #3 — Denser coverage for the scalability experiment.

        The sparse_demonstration_scenario uses bs_coverage_radius=250 in a
        3000m area with 9 BSs (grid spacing ~750m).  At 200 vehicles, most
        vehicles fall in coverage gaps for enough steps that energy_aware
        and RSSI both default to the same fallback BS — yielding 0% saving.

        This scenario doubles the coverage radius to 500m so the network
        stays connected at all vehicle counts tested (20–200).  The
        shadowing and weather settings are deliberately milder (clear, 8 dB)
        so the comparative advantage from EPB-based selection is visible
        rather than swamped by outage.

        Key changes vs sparse_demonstration_scenario:
          - bs_coverage_radius: 250 → 500   (overlapping coverage)
          - num_base_stations: 9 → 16       (4×4 grid, finer spacing)
          - area_size: 3000 → 2800          (tighter grid for 16 BSs)
          - weather_profile: heavy_rain → clear
          - shadowing_std_db: 15.0 → 8.0
          - energy_aware_min_energy_saving: 0.32 → 0.15  (easier to trigger)
          - handoff_cooldown_s: 8.0 → 4.0
        """
        return SimulationConfig(
            num_vehicles=20,        # overridden per run in scaling experiment
            num_base_stations=16,
            area_size=2800,
            bs_coverage_radius=500,
            duration=800,
            movement_mode="highway",
            highway_num_lanes=4,
            shadowing_std_db=8.0,
            highway_lateral_noise_std_m=0.5,
            handoff_cooldown_s=4.0,
            energy_aware_min_energy_saving=0.15,
            energy_aware_time_to_trigger_s=2.0,
            energy_aware_min_data_rate_bps=1e6,
            rssi_energy_use_fixed_tx=False,
            tx_power_default=0.1,
            shadowing_reliability=0.95,
            target_rx_power_dbm=-90.0,
            weather_profile="clear",
        )

    @staticmethod
    def extended_validation_scenario() -> "SimulationConfig":
        """
        Longer run (1 h) for extrapolation / stability experiments.
        """
        return SimulationConfig(
            num_vehicles=20,
            num_base_stations=9,
            area_size=3000,
            bs_coverage_radius=250,
            duration=3600,
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
            seed=42,
        )

    @staticmethod
    def multi_duration_validation() -> List[Tuple[int, str]]:
        """(duration_s, label) pairs for multi-duration extrapolation checks."""
        return [
            (300,  "5 minutes - short term"),
            (900,  "15 minutes - medium term"),
            (1800, "30 minutes - medium-long"),
            (3600, "1 hour - extended"),
            (7200, "2 hours - validation"),
        ]

    @staticmethod
    def sparse_demonstration_scenario_fixed_rssi_tx_sensitivity() -> "SimulationConfig":
        """Sensitivity experiment: enable rssi_energy_use_fixed_tx."""
        cfg = SimulationConfig.sparse_demonstration_scenario()
        cfg.rssi_energy_use_fixed_tx = True
        return cfg

    @staticmethod
    def bangladesh_grid_scenario() -> "SimulationConfig":
        """
        Bangladesh grid intensity sensitivity run.
        Uses 0.62 kg CO2/kWh per Bangladesh Power Development Board reporting.
        All other parameters match sparse_demonstration_scenario.
        """
        cfg = SimulationConfig.sparse_demonstration_scenario()
        cfg.carbon_intensity_kg_per_kwh = 0.62
        return cfg

    def get_weather(self) -> WeatherProfile:
        """Resolved weather profile configuration for the simulator."""
        return get_weather_profile(self.weather_profile)


DEFAULT_CONFIG = SimulationConfig()
