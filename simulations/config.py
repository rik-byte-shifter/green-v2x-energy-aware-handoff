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
    # Default 0.5 = global average.  Bangladesh grid ~= 0.62 kg/kWh.
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
        Main paper baseline comparison scenario.

        FIX -- overlapping coverage so baselines differentiate.
        -------------------------------------------------------
        The original scenario used area_size=3000, 9 BSs, bs_coverage_radius=250.
        The BS grid spacing is 3000//(sqrt(9)+1) = 750m.  With heavy-rain
        shadowing at 15 dB, the 4 highway lane centers sit at y~=1496-1506m,
        essentially on top of the middle row of BSs (y=1500m).  Vehicles
        therefore see AT MOST ONE BS in range at any step:

            lane_y ~= 1498m -> BS at y=1500m: distance 2m  + (in 250m range)
            lane_y ~= 1498m -> BS at y= 750m: distance 748m x
            lane_y ~= 1498m -> BS at y=2250m: distance 752m x

        With only one candidate BS visible, SINR-based and load-aware RSSI
        selection always picks the same BS as RSSI, producing identical EPB
        and CO2 results -- not a comparison failure, just a scenario with no
        choice to make.

        FIX: switch to 16 BSs (4x4 grid) in the same 3000m area.
        New BS grid spacing = 3000//(4+1) = 600m.
        BS y positions: 600, 1200, 1800, 2400.
        Distance from lane centre (y~=1500) to nearest two BSs:
            BS y=1200: 300m  + (within 400m radius)
            BS y=1800: 300m  + (within 400m radius)

        Vehicles now see 2 candidate BSs simultaneously along the x-axis as
        well (BS spacing 600m, radius 400m -> overlap of 200m).  This gives
        all algorithms a genuine choice between candidates that differ in
        SINR, load, and EPB -- exactly what is needed for meaningful comparison.

        Other changes:
          num_base_stations : 9  -> 16   (4x4 grid)
          bs_coverage_radius: 250 -> 400  (overlapping, 2+ candidates/vehicle)
          shadowing_std_db  : 15.0 -> 10.0  (less extreme -> more connected steps)
          handoff_cooldown_s: 8.0 -> 5.0
          energy_aware_min_energy_saving: 0.32 -> 0.20
        Heavy-rain weather profile is retained to preserve adverse-condition
        framing for the IEES paper.
        """
        return SimulationConfig(
            num_vehicles=20,
            num_base_stations=16,
            area_size=3000,
            bs_coverage_radius=400,
            duration=800,
            movement_mode="highway",
            highway_num_lanes=4,
            shadowing_std_db=10.0,
            highway_lateral_noise_std_m=0.5,
            handoff_cooldown_s=5.0,
            energy_aware_min_energy_saving=0.20,
            energy_aware_time_to_trigger_s=3.0,
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
        Scalability experiment: denser network so 200-vehicle runs stay connected.

        Uses 16 BSs, radius=500, clear weather so algorithm differences are
        visible at all vehicle counts (20-200) without coverage-gap collapse.
        max_capacity=100 in BSConfig means has_capacity() stays True throughout.
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
        Same overlapping coverage as sparse_demonstration_scenario.
        """
        return SimulationConfig(
            num_vehicles=20,
            num_base_stations=16,
            area_size=3000,
            bs_coverage_radius=400,
            duration=3600,
            movement_mode="highway",
            highway_num_lanes=4,
            shadowing_std_db=10.0,
            highway_lateral_noise_std_m=0.5,
            handoff_cooldown_s=5.0,
            energy_aware_min_energy_saving=0.20,
            energy_aware_time_to_trigger_s=3.0,
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
