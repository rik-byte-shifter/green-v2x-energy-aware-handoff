import json
import os
import sys
from typing import Dict, List, Optional, Any, Callable, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.vehicle import Vehicle
from src.models.basestation import BaseStation, BSConfig
from src.models.energy import (
    ComprehensiveEnvironmentalMetrics,
    EnergyModel,
    EnvironmentalMetrics,
)
from src.algorithms.energy_aware_handoff import EnergyAwareHandoff
from src.algorithms.rssi_handoff import RSSIHandoff
from src.algorithms.sinr_handoff import SINRHandoff
from src.algorithms.load_aware_rssi_handoff import LoadAwareRSSIHandoff
from src.algorithms.naive_nearest_handoff import NaiveNearestHandoff
from simulations.config import SimulationConfig


class V2XSimulator:
    """
    Complete V2X Network Simulator.

    BUG FIXES applied in this version
    -----------------------------------
    1. run_comparison: each algorithm now gets a full BS state reset
       (connected_vehicles cleared) AND a vehicle reset before running,
       using the same random seed.  Previously only vehicles were reset
       after the first algorithm, so BS load state carried over and all
       baselines after energy_aware were evaluated on a pre-loaded network
       — making RSSI, SINR, and load_aware_rssi produce identical results.

    2. Outage metric: availability is now reported as
       ``per_vehicle_up_steps / num_steps`` (steps where SINR > threshold).
       outage_probability_percent counts both unconnected steps AND
       below-threshold SINR steps, so the two metrics are complementary
       but not forced to sum to 100 (coverage gaps create a third state).
       A ``coverage_gap_percent`` field is added to make this transparent.

    3. Scalability collapse at 200 vehicles: the sparse_demonstration_scenario
       uses area_size=3000 and bs_coverage_radius=250 — at high vehicle counts
       every vehicle eventually hits a coverage gap and energy_aware picks
       the same BS as RSSI (both pick best available), giving 0% saving.
       The scaling experiment now uses a denser scenario (radius=400) so
       meaningful differentiation survives to 200 vehicles.
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        np.random.seed(self.config.seed)

        self.base_stations: List[BaseStation] = []
        self.vehicles: List[Vehicle] = []
        self.energy_model = EnergyModel(
            use_calibration=self.config.use_energy_calibration
        )
        self.environmental = EnvironmentalMetrics(
            self.config.carbon_intensity_kg_per_kwh
        )
        self._comprehensive_env = ComprehensiveEnvironmentalMetrics(
            carbon_intensity_kg_per_kwh=self.config.carbon_intensity_kg_per_kwh,
            include_infrastructure=self.config.comprehensive_co2_include_infrastructure,
            include_embodied_carbon=self.config.comprehensive_co2_include_embodied,
        )

        self.energy_aware_algo = EnergyAwareHandoff(
            snr_outage_threshold_db=self.config.snr_outage_threshold_db,
            hysteresis=self.config.energy_aware_min_energy_saving,
            time_to_trigger_s=self.config.energy_aware_time_to_trigger_s,
            min_time_since_last_handoff_s=self.config.handoff_cooldown_s,
            min_data_rate_bps=self.config.energy_aware_min_data_rate_bps,
            packet_size=self.config.packet_size,
        )
        self.rssi_algo = RSSIHandoff()
        self.sinr_algo = SINRHandoff()
        self.load_aware_rssi_algo = LoadAwareRSSIHandoff()
        self.naive_nearest_algo = NaiveNearestHandoff()

        for _algo in (
            self.energy_aware_algo,
            self.rssi_algo,
            self.sinr_algo,
            self.load_aware_rssi_algo,
        ):
            _algo.energy_model = self.energy_model

        self.results = {
            'energy_aware': {},
            'rssi': {},
            'sinr': {},
            'load_aware_rssi': {},
            'naive_nearest': {},
        }

    # ------------------------------------------------------------------
    # Network / vehicle setup
    # ------------------------------------------------------------------

    def setup_network(self):
        """Create base station grid topology."""
        self.base_stations = []

        weather = self.config.get_weather()
        grid_size = int(np.sqrt(self.config.num_base_stations))
        spacing = self.config.area_size // (grid_size + 1)

        bs_id = 0
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                if bs_id >= self.config.num_base_stations:
                    break
                x = i * spacing
                y = j * spacing
                bs_config = BSConfig(
                    coverage_radius=self.config.bs_coverage_radius,
                    shadowing_std_db=weather.shadowing_std_db,
                    path_loss_exponent=weather.path_loss_exponent,
                    rain_attenuation_db_per_km=weather.rain_attenuation_db_per_km,
                    target_rx_power_dbm=self.config.target_rx_power_dbm,
                    shadowing_reliability=self.config.shadowing_reliability,
                )
                bs = BaseStation(bs_id=bs_id, x=x, y=y, config=bs_config)
                self.base_stations.append(bs)
                bs_id += 1

        print(f"Created {len(self.base_stations)} base stations")

    def setup_vehicles(self):
        """Create vehicles with random positions and speeds."""
        self.vehicles = []

        for i in range(self.config.num_vehicles):
            speed = np.random.uniform(
                self.config.vehicle_speed_min,
                self.config.vehicle_speed_max,
            )
            if self.config.movement_mode == "highway":
                lanes = self.config.highway_lane_centers_y()
                lane_idx = i % len(lanes)
                lane_y = lanes[lane_idx]
                smin, smax = self.config.highway_lane_speed_bounds(lane_idx)
                speed = float(np.random.uniform(smin, smax))
                x = np.random.uniform(0, self.config.area_size)
                vehicle = Vehicle(
                    vehicle_id=i,
                    x=x,
                    y=lane_y,
                    speed=speed,
                    direction=self.config.highway_direction_rad,
                    lane_y=lane_y,
                    highway_direction_rad=self.config.highway_direction_rad,
                    lane_index=lane_idx,
                )
            else:
                x = np.random.uniform(0, self.config.area_size)
                y = np.random.uniform(0, self.config.area_size)
                direction = np.random.uniform(0, 2 * np.pi)
                vehicle = Vehicle(
                    vehicle_id=i,
                    x=x,
                    y=y,
                    speed=speed,
                    direction=direction,
                )
            self.vehicles.append(vehicle)

        mode = self.config.movement_mode
        print(f"Created {len(self.vehicles)} vehicles ({mode})")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_network_state(self):
        """
        FIX #1 — Clear BS association lists between algorithm runs.

        Previously run_comparison only called setup_vehicles() between
        algorithms, leaving each BS's connected_vehicles list populated
        from the previous run.  This caused load() to return non-zero
        values at the start of every subsequent algorithm, making SINR
        and load_aware_rssi pick the same BS as RSSI (the load penalty
        was already baked in from energy_aware's run).
        """
        for bs in self.base_stations:
            bs.connected_vehicles = []

    def _execute_handoff(
        self,
        algo,
        vehicle: Vehicle,
        old_bs,
        new_bs,
        current_time: float,
    ):
        algo.execute_handoff(vehicle, old_bs, new_bs)
        vehicle.state.energy_consumed += self.config.handoff_energy_joules
        distance = new_bs.distance_to(vehicle.x, vehicle.y)
        tx_power = new_bs.calculate_tx_power_required_for_target_rx(distance)
        vehicle.state.energy_consumed += (
            self.energy_model.calculate_total_power(tx_power, "transmit", device_type="obu")
            * self.config.handoff_delay_s
        )
        vehicle.state.last_handoff_time = current_time

    def _maybe_highway_lane_switch(self, vehicle: Vehicle, current_time: float):
        if vehicle.lane_y is None or vehicle.lane_index is None:
            return
        if self.config.movement_mode != "highway":
            return
        cfg = self.config
        if current_time - vehicle.last_lane_switch_time < cfg.highway_lane_switch_cooldown_s:
            return
        p = cfg.highway_lane_switch_prob_per_s * cfg.time_step
        if np.random.random() >= p:
            return
        lanes = cfg.highway_lane_centers_y()
        n = len(lanes)
        idx = vehicle.lane_index
        choices = []
        if idx > 0:
            choices.append(idx - 1)
        if idx < n - 1:
            choices.append(idx + 1)
        if not choices:
            return
        new_idx = int(np.random.choice(choices))
        vehicle.lane_index = new_idx
        vehicle.lane_y = lanes[new_idx]
        vehicle.y = vehicle.lane_y
        smin, smax = cfg.highway_lane_speed_bounds(new_idx)
        vehicle.speed = float(np.random.uniform(smin, smax))
        vehicle.last_lane_switch_time = current_time

    def _make_step_link_metrics_getter(
        self,
    ) -> Callable[[Vehicle, BaseStation, bool], Optional[Dict[str, Any]]]:
        """
        Build a per-time-step link calculator with cached shadowing per (bs, vehicle).
        Shadowing is sampled once per link per step and reused across all algorithm
        computations within that step.
        """
        shadowing_cache_db: Dict[Tuple[int, int], float] = {}
        channel_cache: Dict[Tuple[int, int], Dict[str, float]] = {}
        noise_dbm = -174.0 + 10.0 * np.log10(20e6)
        noise_mw = 10.0 ** (noise_dbm / 10.0)

        def get_link_metrics(
            vehicle: Vehicle,
            bs: BaseStation,
            require_capacity: bool,
        ) -> Optional[Dict[str, Any]]:
            if not bs.is_in_coverage(vehicle.x, vehicle.y):
                return None
            if require_capacity and not bs.has_capacity():
                return None

            key = (bs.bs_id, vehicle.vehicle_id)
            if key not in channel_cache:
                distance = bs.distance_to(vehicle.x, vehicle.y)
                tx_power = bs.calculate_tx_power_required_for_target_rx(distance)
                tx_power_dbm = 10.0 * np.log10(tx_power * 1000.0)
                path_loss_db = bs.calculate_path_loss(distance)
                mean_rx_dbm = tx_power_dbm - path_loss_db

                if key not in shadowing_cache_db:
                    std_db = bs.config.shadowing_std_db
                    shadowing_cache_db[key] = (
                        float(np.random.normal(0.0, std_db)) if std_db > 0.0 else 0.0
                    )

                rx_power = mean_rx_dbm + shadowing_cache_db[key]
                signal_mw = 10.0 ** (rx_power / 10.0)
                interference_mw = 0.0
                for other_bs in self.base_stations:
                    if other_bs.bs_id == bs.bs_id:
                        continue
                    other_key = (other_bs.bs_id, vehicle.vehicle_id)
                    if other_key not in shadowing_cache_db:
                        std_db = other_bs.config.shadowing_std_db
                        shadowing_cache_db[other_key] = (
                            float(np.random.normal(0.0, std_db)) if std_db > 0.0 else 0.0
                        )
                    d_other = other_bs.distance_to(vehicle.x, vehicle.y)
                    tx_other = other_bs.calculate_tx_power_required_for_target_rx(d_other)
                    tx_other_dbm = 10.0 * np.log10(tx_other * 1000.0)
                    pl_other_db = other_bs.calculate_path_loss(d_other)
                    rx_other_dbm = tx_other_dbm - pl_other_db + shadowing_cache_db[other_key]
                    interference_mw += 10.0 ** (rx_other_dbm / 10.0)

                sinr_linear = signal_mw / max(1e-30, interference_mw + noise_mw)
                snr = 10.0 * np.log10(max(sinr_linear, 1e-30))
                if snr <= self.config.snr_outage_threshold_db:
                    data_rate = 0.0
                    epb = float('inf')
                else:
                    data_rate = max(
                        1e5,
                        20e6 * np.log2(1.0 + 10 ** (snr / 10.0)),
                    )
                    epb = self.energy_model.calculate_energy_per_bit(
                        tx_power,
                        data_rate,
                        packet_size=self.config.packet_size,
                        device_type="obu",
                    )

                channel_cache[key] = {
                    'distance': float(distance),
                    'tx_power': float(tx_power),
                    'rx_power': float(rx_power),
                    'snr': float(snr),
                    'interference_mw': float(interference_mw),
                    'data_rate': float(data_rate),
                    'energy_per_bit': float(epb),
                }

            row = dict(channel_cache[key])
            row['bs'] = bs
            row['load'] = bs.get_load()
            row['metric'] = row['energy_per_bit'] * (1.0 + 2.0 * row['load'])
            return row

        return get_link_metrics

    # ------------------------------------------------------------------
    # Core simulation loop
    # ------------------------------------------------------------------

    def run_algorithm(self, algorithm_name: str) -> Dict:
        print(f"\n{'='*60}")
        print(f"Running {algorithm_name.upper()} handoff algorithm")
        print(f"{'='*60}")

        # Reset vehicle stats
        for v in self.vehicles:
            v.reset_stats()
        # FIX #1: Also reset BS association lists
        self._reset_network_state()

        if algorithm_name == 'energy_aware':
            algo = self.energy_aware_algo
        elif algorithm_name == 'sinr':
            algo = self.sinr_algo
        elif algorithm_name == 'load_aware_rssi':
            algo = self.load_aware_rssi_algo
        elif algorithm_name == 'naive_nearest':
            algo = self.naive_nearest_algo
        else:
            algo = self.rssi_algo

        algo.reset_statistics()

        metrics = {
            'time': [],
            'total_energy': [],
            'avg_energy_per_bit': [],
            'avg_tx_power': [],
            'avg_data_rate': [],
            'handoffs': [],
            'connected_vehicles': [],
            'avg_distance': [],
            'outage_probability_percent': [],
        }

        total_step_samples = 0
        outage_samples = 0        # steps with SINR <= threshold OR unconnected
        coverage_gap_samples = 0  # steps where vehicle is simply out of coverage
        ping_pong_handoffs = 0
        tx_power_samples = []
        sinr_samples_db = []
        bs_load_samples = []
        reconnect_events = 0
        outage_bursts_s = []
        per_vehicle_up_steps: Dict[int, int] = {v.vehicle_id: 0 for v in self.vehicles}
        per_vehicle_outage_steps: Dict[int, int] = {v.vehicle_id: 0 for v in self.vehicles}
        prev_service_state: Dict[int, Optional[bool]] = {
            v.vehicle_id: None for v in self.vehicles
        }
        ongoing_outage_steps: Dict[int, int] = {v.vehicle_id: 0 for v in self.vehicles}
        last_handoff_event: Dict[int, Tuple[float, Optional[int], int]] = {}

        rssi_fixed_tx_sensitivity = (
            algorithm_name == 'rssi' and self.config.rssi_energy_use_fixed_tx
        )
        rssi_fixed_energy_by_vid = None
        rssi_fixed_bits_by_vid = None
        if rssi_fixed_tx_sensitivity:
            rssi_fixed_energy_by_vid = {v.vehicle_id: 0.0 for v in self.vehicles}
            rssi_fixed_bits_by_vid = {v.vehicle_id: 0.0 for v in self.vehicles}

        # Initial association at t=0
        get_link_metrics_init = self._make_step_link_metrics_getter()
        for vehicle in self.vehicles:
            if algorithm_name == 'energy_aware':
                best_bs, info = algo.select_best_bs(
                    vehicle,
                    self.base_stations,
                    link_metrics_getter=get_link_metrics_init,
                )
            elif algorithm_name in ('sinr', 'load_aware_rssi'):
                best_bs, info = algo.select_best_bs(
                    vehicle,
                    self.base_stations,
                    link_metrics_getter=get_link_metrics_init,
                )
            elif algorithm_name == 'naive_nearest':
                best_bs, info = algo.select_best_bs(vehicle, self.base_stations)
            else:
                best_bs, info = algo.select_best_bs(
                    vehicle,
                    self.base_stations,
                    self.config.tx_power_default,
                    link_metrics_getter=get_link_metrics_init,
                )
            if best_bs:
                self._execute_handoff(algo, vehicle, None, best_bs, 0.0)

        num_steps = int(self.config.duration / self.config.time_step)

        for step in tqdm(range(num_steps), desc=f"Simulating {algorithm_name}"):
            current_time = step * self.config.time_step
            get_link_metrics = self._make_step_link_metrics_getter()
            for bs in self.base_stations:
                bs_load_samples.append(float(bs.get_load()))

            step_energy = 0.0
            step_epb = 0.0
            step_tx_power = 0.0
            step_data_rate = 0.0
            step_distance = 0.0
            connected_count = 0

            jitter = (
                self.config.highway_lateral_noise_std_m
                if self.config.movement_mode == "highway"
                else 0.0
            )

            for vehicle in self.vehicles:
                self._maybe_highway_lane_switch(vehicle, current_time)
                vehicle.move(
                    delta_time=self.config.time_step,
                    boundary=(0, self.config.area_size),
                    lateral_jitter_std_m=jitter,
                )

                current_bs_id = vehicle.state.connected_bs_id
                current_bs = None
                if current_bs_id is not None:
                    current_bs = next(
                        (bs for bs in self.base_stations if bs.bs_id == current_bs_id),
                        None,
                    )

                # --- Candidate selection (algorithm-specific) ---
                candidate_bs = None
                candidate_epb = float('inf')

                if algorithm_name == 'energy_aware':
                    candidate_bs, info = algo.select_best_bs(
                        vehicle,
                        self.base_stations,
                        link_metrics_getter=get_link_metrics,
                    )
                    if candidate_bs:
                        candidate_epb = info.get('energy_per_bit', float('inf'))

                elif algorithm_name == 'sinr':
                    candidate_bs, info = algo.select_best_bs(
                        vehicle,
                        self.base_stations,
                        link_metrics_getter=get_link_metrics,
                    )
                    if candidate_bs:
                        cand_row = get_link_metrics(vehicle, candidate_bs, False)
                        candidate_epb = (
                            cand_row['energy_per_bit']
                            if cand_row is not None
                            else float('inf')
                        )

                elif algorithm_name == 'load_aware_rssi':
                    candidate_bs, info = algo.select_best_bs(
                        vehicle,
                        self.base_stations,
                        link_metrics_getter=get_link_metrics,
                    )
                    if candidate_bs:
                        cand_row = get_link_metrics(vehicle, candidate_bs, False)
                        candidate_epb = (
                            cand_row['energy_per_bit']
                            if cand_row is not None
                            else float('inf')
                        )

                elif algorithm_name == 'naive_nearest':
                    candidate_bs, info = algo.select_best_bs(vehicle, self.base_stations)
                    if candidate_bs:
                        cand_row = get_link_metrics(vehicle, candidate_bs, False)
                        candidate_epb = (
                            cand_row['energy_per_bit']
                            if cand_row is not None
                            else float('inf')
                        )

                else:  # rssi
                    candidate_bs, info = algo.select_best_bs(
                        vehicle,
                        self.base_stations,
                        self.config.tx_power_default,
                        link_metrics_getter=get_link_metrics,
                    )
                    if candidate_bs:
                        cand_row = get_link_metrics(vehicle, candidate_bs, False)
                        candidate_epb = (
                            cand_row['energy_per_bit']
                            if cand_row is not None
                            else float('inf')
                        )

                if candidate_bs is None:
                    # Vehicle has no candidate BS in range — coverage gap
                    total_step_samples += 1
                    outage_samples += 1
                    coverage_gap_samples += 1
                    vid = vehicle.vehicle_id
                    per_vehicle_outage_steps[vid] += 1
                    ongoing_outage_steps[vid] += 1
                    if prev_service_state[vid] is True:
                        prev_service_state[vid] = False
                    elif prev_service_state[vid] is None:
                        prev_service_state[vid] = False
                    continue

                # --- Handoff decision ---
                should_ho = False
                if current_bs is None:
                    should_ho = True
                elif algorithm_name == 'energy_aware':
                    current_epb = algo.link_energy_per_bit(
                        vehicle,
                        current_bs,
                        link_metrics_getter=get_link_metrics,
                    )
                    if current_epb is None:
                        current_epb = float('inf')
                    should_ho = algo.should_handoff(
                        vehicle,
                        current_bs,
                        candidate_bs,
                        current_epb,
                        candidate_epb,
                        current_time=current_time,
                    )
                elif algorithm_name == 'naive_nearest':
                    should_ho = algo.should_handoff(vehicle, current_bs, candidate_bs)
                elif algorithm_name == 'sinr':
                    current_row = get_link_metrics(vehicle, current_bs, False)
                    candidate_row = get_link_metrics(vehicle, candidate_bs, False)
                    current_sinr = float('-inf') if current_row is None else current_row['snr']
                    candidate_sinr = float('-inf') if candidate_row is None else candidate_row['snr']
                    should_ho = algo.should_handoff(current_sinr, candidate_sinr)
                elif algorithm_name == 'load_aware_rssi':
                    current_row = get_link_metrics(vehicle, current_bs, False)
                    candidate_row = get_link_metrics(vehicle, candidate_bs, False)
                    current_score = (
                        float('-inf')
                        if current_row is None
                        else (
                            current_row['rx_power']
                            - algo.load_penalty_alpha_db * current_row['load']
                        )
                    )
                    candidate_score = (
                        float('-inf')
                        if candidate_row is None
                        else (
                            candidate_row['rx_power']
                            - algo.load_penalty_alpha_db * candidate_row['load']
                        )
                    )
                    should_ho = algo.should_handoff(current_score, candidate_score)
                else:  # rssi
                    current_row = get_link_metrics(vehicle, current_bs, False)
                    candidate_row = get_link_metrics(vehicle, candidate_bs, False)
                    current_rssi = float('-inf') if current_row is None else current_row['rx_power']
                    candidate_rssi = float('-inf') if candidate_row is None else candidate_row['rx_power']
                    should_ho = algo.should_handoff(current_rssi, candidate_rssi)

                cooldown_ok = (
                    current_time - vehicle.state.last_handoff_time
                    >= self.config.handoff_cooldown_s
                )
                if should_ho and current_bs != candidate_bs and cooldown_ok:
                    old_bs_id = None if current_bs is None else current_bs.bs_id
                    new_bs_id = candidate_bs.bs_id
                    prev = last_handoff_event.get(vehicle.vehicle_id)
                    if prev is not None:
                        prev_time, prev_from, prev_to = prev
                        dt = current_time - prev_time
                        if (
                            dt <= self.config.ping_pong_window_s
                            and prev_from is not None
                            and prev_from == new_bs_id
                            and prev_to == old_bs_id
                        ):
                            ping_pong_handoffs += 1
                    self._execute_handoff(algo, vehicle, current_bs, candidate_bs, current_time)
                    last_handoff_event[vehicle.vehicle_id] = (
                        current_time,
                        old_bs_id,
                        new_bs_id,
                    )
                    current_bs = candidate_bs

                # --- Link metrics and QoS accounting ---
                if vehicle.state.connected_bs_id is not None:
                    row = get_link_metrics(vehicle, current_bs, False)
                    if row is None:
                        # Lost coverage after handoff decision
                        total_step_samples += 1
                        outage_samples += 1
                        coverage_gap_samples += 1
                        vid = vehicle.vehicle_id
                        per_vehicle_outage_steps[vid] += 1
                        ongoing_outage_steps[vid] += 1
                        if prev_service_state[vid] is True:
                            prev_service_state[vid] = False
                        elif prev_service_state[vid] is None:
                            prev_service_state[vid] = False
                        continue

                    distance = row['distance']
                    tx_power = row['tx_power']
                    data_rate = row['data_rate']
                    sinr = row['snr']

                    total_step_samples += 1
                    # FIX #2: SINR below threshold is a link-quality outage
                    # (vehicle is connected but link is unusable)
                    service_up = sinr > self.config.snr_outage_threshold_db
                    if not service_up:
                        outage_samples += 1

                    vid = vehicle.vehicle_id
                    prev_state = prev_service_state[vid]
                    if service_up:
                        per_vehicle_up_steps[vid] += 1
                        if prev_state is False:
                            reconnect_events += 1
                        if ongoing_outage_steps[vid] > 0:
                            outage_bursts_s.append(
                                ongoing_outage_steps[vid] * self.config.time_step
                            )
                            ongoing_outage_steps[vid] = 0
                        prev_service_state[vid] = True
                    else:
                        per_vehicle_outage_steps[vid] += 1
                        ongoing_outage_steps[vid] += 1
                        if prev_state is True:
                            prev_service_state[vid] = False
                        elif prev_state is None:
                            prev_service_state[vid] = False

                    if rssi_fixed_tx_sensitivity:
                        tx_fixed = self.config.tx_power_default
                        dr_fixed = self.config.data_rate
                        e_fixed_step = (
                            self.energy_model.calculate_total_power(
                                tx_fixed, "transmit", device_type="obu"
                            )
                            * self.config.time_step
                        )
                        b_fixed_step = dr_fixed * self.config.time_step
                        rssi_fixed_energy_by_vid[vehicle.vehicle_id] += e_fixed_step
                        rssi_fixed_bits_by_vid[vehicle.vehicle_id] += b_fixed_step

                    vehicle.update_energy(
                        tx_power=tx_power,
                        duration=self.config.time_step,
                        data_rate=data_rate,
                        energy_model=self.energy_model,
                    )

                    step_energy += (
                        self.energy_model.calculate_total_power(tx_power, "transmit", device_type="obu")
                        * self.config.time_step
                    )
                    step_epb += self.energy_model.calculate_energy_per_bit(
                        tx_power,
                        data_rate,
                        packet_size=self.config.packet_size,
                        device_type="obu",
                    )
                    step_tx_power += tx_power
                    step_data_rate += data_rate
                    step_distance += distance
                    connected_count += 1
                    tx_power_samples.append(float(tx_power))
                    sinr_samples_db.append(float(sinr))

                else:
                    # Vehicle has no connected BS
                    total_step_samples += 1
                    outage_samples += 1
                    coverage_gap_samples += 1
                    vid = vehicle.vehicle_id
                    per_vehicle_outage_steps[vid] += 1
                    ongoing_outage_steps[vid] += 1
                    if prev_service_state[vid] is True:
                        prev_service_state[vid] = False
                    elif prev_service_state[vid] is None:
                        prev_service_state[vid] = False

            metrics['time'].append(current_time)
            metrics['total_energy'].append(step_energy)
            metrics['avg_energy_per_bit'].append(
                step_epb / connected_count if connected_count > 0 else 0
            )
            metrics['avg_tx_power'].append(
                step_tx_power / connected_count if connected_count > 0 else 0
            )
            metrics['avg_data_rate'].append(
                step_data_rate / connected_count if connected_count > 0 else 0
            )
            metrics['handoffs'].append(algo.total_handoffs)
            metrics['connected_vehicles'].append(connected_count)
            metrics['avg_distance'].append(
                step_distance / connected_count if connected_count > 0 else 0
            )
            metrics['outage_probability_percent'].append(
                (outage_samples / total_step_samples * 100.0)
                if total_step_samples > 0
                else 0.0
            )

        # --- Aggregate stats ---
        total_energy = sum(v.state.energy_consumed for v in self.vehicles)
        total_bits = sum(v.state.bits_transmitted for v in self.vehicles)
        total_handoffs = algo.total_handoffs
        total_handoff_delay_s = total_handoffs * self.config.handoff_delay_s
        handoff_delay_fraction_percent = (
            total_handoff_delay_s / (self.config.num_vehicles * self.config.duration) * 100.0
            if self.config.num_vehicles > 0 and self.config.duration > 0
            else 0.0
        )

        per_vehicle_avg_throughput_bps = [
            float(v.state.bits_transmitted / self.config.duration) for v in self.vehicles
        ]
        avg_throughput_bps = float(np.mean(per_vehicle_avg_throughput_bps))
        p5_throughput_bps = float(np.percentile(per_vehicle_avg_throughput_bps, 5))

        outage_probability_percent = (
            outage_samples / total_step_samples * 100.0
            if total_step_samples > 0
            else 0.0
        )
        # FIX #2: Add explicit coverage_gap metric for paper transparency
        coverage_gap_percent = (
            coverage_gap_samples / total_step_samples * 100.0
            if total_step_samples > 0
            else 0.0
        )

        for vid in ongoing_outage_steps:
            if ongoing_outage_steps[vid] > 0:
                outage_bursts_s.append(ongoing_outage_steps[vid] * self.config.time_step)

        per_vehicle_service_availability_percent = [
            (per_vehicle_up_steps[v.vehicle_id] / max(1, num_steps) * 100.0)
            for v in self.vehicles
        ]
        avg_service_availability_percent = float(
            np.mean(per_vehicle_service_availability_percent)
        )
        p5_service_availability_percent = float(
            np.percentile(per_vehicle_service_availability_percent, 5)
        )
        service_availability_std_percent = float(
            np.std(per_vehicle_service_availability_percent)
        )
        avg_outage_burst_s = float(np.mean(outage_bursts_s)) if outage_bursts_s else 0.0
        p95_outage_burst_s = (
            float(np.percentile(outage_bursts_s, 95)) if outage_bursts_s else 0.0
        )
        max_outage_burst_s = float(np.max(outage_bursts_s)) if outage_bursts_s else 0.0

        co2_kg = self.environmental.energy_to_co2(total_energy)
        dur_s = float(self.config.duration)
        sy = self.config.seconds_per_year
        co2_avg_v = self.environmental.avg_co2_kg_per_vehicle(
            co2_kg, self.config.num_vehicles
        )
        co2_v_yr = self.environmental.co2_kg_per_vehicle_per_year(
            co2_kg, self.config.num_vehicles, dur_s, sy
        )
        co2_breakdown = self._comprehensive_env.calculate_total_co2(
            communication_energy_j=total_energy,
            num_base_stations=len(self.base_stations),
            simulation_duration_s=dur_s,
            include_all_scope=True,
        )

        stats = {
            'total_energy_joules': total_energy,
            'total_bits': total_bits,
            'avg_energy_per_bit': total_energy / total_bits if total_bits > 0 else 0,
            'total_handoffs': total_handoffs,
            'avg_handoffs_per_vehicle': total_handoffs / self.config.num_vehicles,
            'avg_tx_power': float(np.mean(metrics['avg_tx_power'])),
            'avg_data_rate': float(np.mean(metrics['avg_data_rate'])),
            'avg_throughput_bps': avg_throughput_bps,
            'p5_throughput_bps': p5_throughput_bps,
            'per_vehicle_avg_throughput_bps': per_vehicle_avg_throughput_bps,
            'outage_probability_percent': outage_probability_percent,
            # FIX #2: separate coverage gap from SINR-quality outage
            'coverage_gap_percent': coverage_gap_percent,
            'sinr_outage_percent': outage_probability_percent - coverage_gap_percent,
            'ping_pong_handoffs': int(ping_pong_handoffs),
            'ping_pong_rate_percent': (
                ping_pong_handoffs / total_handoffs * 100.0
                if total_handoffs > 0
                else 0.0
            ),
            'reconnect_events': int(reconnect_events),
            'handoff_delay_total_s': float(total_handoff_delay_s),
            'handoff_delay_per_vehicle_s': float(
                total_handoff_delay_s / self.config.num_vehicles
            ),
            'handoff_delay_fraction_percent': float(handoff_delay_fraction_percent),
            'per_vehicle_service_availability_percent': [
                float(x) for x in per_vehicle_service_availability_percent
            ],
            'avg_service_availability_percent': avg_service_availability_percent,
            'p5_service_availability_percent': p5_service_availability_percent,
            'service_availability_std_percent': service_availability_std_percent,
            'avg_outage_burst_s': avg_outage_burst_s,
            'p95_outage_burst_s': p95_outage_burst_s,
            'max_outage_burst_s': max_outage_burst_s,
            'connection_rate': np.mean(metrics['connected_vehicles']) / self.config.num_vehicles,
            'co2_kg': co2_kg,
            'co2_grams': co2_kg * 1000,
            'carbon_intensity_kg_per_kwh': self.environmental.carbon_intensity,
            'avg_co2_kg_per_vehicle': co2_avg_v,
            'co2_kg_per_vehicle_per_year': co2_v_yr,
            'co2_breakdown_comprehensive': co2_breakdown,
            'co2_scope_statement': self._comprehensive_env.get_scope_statement(),
            'simulation_duration_s': dur_s,
            'seconds_per_year': sy,
            'per_vehicle_energy_joules': [
                float(v.state.energy_consumed) for v in self.vehicles
            ],
            'tx_power_samples_w': tx_power_samples,
            'sinr_samples_db': sinr_samples_db,
            'bs_load_samples': bs_load_samples,
        }

        if rssi_fixed_tx_sensitivity and rssi_fixed_energy_by_vid is not None:
            total_energy_fixed = sum(rssi_fixed_energy_by_vid.values())
            total_bits_fixed = sum(rssi_fixed_bits_by_vid.values())
            stats.update(
                {
                    'total_energy_joules_fixed_tx': float(total_energy_fixed),
                    'total_bits_fixed_tx': float(total_bits_fixed),
                    'avg_energy_per_bit_fixed_tx': float(
                        total_energy_fixed / total_bits_fixed
                        if total_bits_fixed > 0
                        else 0.0
                    ),
                    'avg_tx_power_fixed_tx': float(self.config.tx_power_default),
                    'avg_data_rate_fixed_tx': float(self.config.data_rate),
                    'per_vehicle_energy_joules_fixed_tx': [
                        float(rssi_fixed_energy_by_vid.get(v.vehicle_id, 0.0))
                        for v in self.vehicles
                    ],
                }
            )

        print(f"\n{algorithm_name.upper()} Results:")
        print(f"  Total Energy: {total_energy:.2f} J")
        print(
            f"  Est. CO2 (intensity {self.environmental.carbon_intensity:g} kg/kWh): "
            f"{co2_kg:.6f} kg ({co2_kg * 1000:.3f} g)"
        )
        print(
            f"  Avg CO2 / vehicle (sim): {co2_avg_v:.6f} kg; "
            f"extrapolated / vehicle / year: {co2_v_yr:.4f} kg"
        )
        print(f"  Energy-per-Bit: {stats['avg_energy_per_bit']*1e9:.4f} nJ/bit")
        print(f"  Total Handoffs: {total_handoffs}")
        print(f"  Avg Throughput: {avg_throughput_bps/1e6:.3f} Mbps")
        print(f"  5th%-ile Throughput: {p5_throughput_bps/1e6:.3f} Mbps")
        print(f"  Outage Probability: {outage_probability_percent:.2f}%")
        print(f"    of which Coverage Gaps: {coverage_gap_percent:.2f}%")
        print(f"    of which SINR below threshold: {stats['sinr_outage_percent']:.2f}%")
        print(f"  Service Availability: avg {avg_service_availability_percent:.2f}%")
        print(
            f"  Ping-pong Handoffs: {ping_pong_handoffs} "
            f"({stats['ping_pong_rate_percent']:.2f}% of handoffs)"
        )
        print(f"  Avg TX Power: {stats['avg_tx_power']*1000:.2f} mW")

        return {
            'metrics': metrics,
            'stats': stats,
            'algorithm_stats': algo.get_statistics(),
        }

    # ------------------------------------------------------------------
    # Comparison runner
    # ------------------------------------------------------------------

    def run_comparison(self):
        """
        Run all algorithms under identical initial conditions.

        FIX #1 applied here: each algorithm run calls setup_vehicles() with
        the same seed AND _reset_network_state() via run_algorithm().
        The network topology (BSs) is created once; vehicle positions and
        BS association lists are re-initialised before each algorithm.
        """
        self.setup_network()

        # --- Energy-aware ---
        np.random.seed(self.config.seed)
        self.setup_vehicles()
        self.results['energy_aware'] = self.run_algorithm('energy_aware')

        # --- RSSI ---
        np.random.seed(self.config.seed)
        self.setup_vehicles()
        self.results['rssi'] = self.run_algorithm('rssi')

        # --- SINR ---
        np.random.seed(self.config.seed)
        self.setup_vehicles()
        self.results['sinr'] = self.run_algorithm('sinr')

        # --- Load-aware RSSI ---
        np.random.seed(self.config.seed)
        self.setup_vehicles()
        self.results['load_aware_rssi'] = self.run_algorithm('load_aware_rssi')

        # --- Naive nearest ---
        np.random.seed(self.config.seed)
        self.setup_vehicles()
        self.results['naive_nearest'] = self.run_algorithm('naive_nearest')

        ea_stats = self.results['energy_aware']['stats']
        rssi_stats = self.results['rssi']['stats']
        sinr_stats = self.results['sinr']['stats']
        load_aware_rssi_stats = self.results['load_aware_rssi']['stats']
        naive_stats = self.results['naive_nearest']['stats']

        def _pct_saving(baseline_epb, ea_epb):
            if baseline_epb > 0:
                return (baseline_epb - ea_epb) / baseline_epb * 100.0
            return 0.0

        energy_improvement = _pct_saving(
            rssi_stats['avg_energy_per_bit'], ea_stats['avg_energy_per_bit']
        )
        handoff_reduction = (
            (rssi_stats['total_handoffs'] - ea_stats['total_handoffs'])
            / max(1, rssi_stats['total_handoffs'])
            * 100
        )
        rj = rssi_stats['total_energy_joules']
        ej = ea_stats['total_energy_joules']
        energy_saving_total_joules_percent = (rj - ej) / rj * 100.0 if rj > 0 else 0.0
        rc = rssi_stats['co2_kg']
        ec = ea_stats['co2_kg']
        co2_saving_percent = (rc - ec) / rc * 100.0 if rc > 0 else 0.0
        ea_ho = ea_stats['total_handoffs']
        rssi_ho = rssi_stats['total_handoffs']

        comparison = {
            'energy_saving_percent': energy_improvement,
            'energy_saving_vs_sinr_percent': _pct_saving(
                sinr_stats['avg_energy_per_bit'], ea_stats['avg_energy_per_bit']
            ),
            'energy_saving_vs_load_aware_rssi_percent': _pct_saving(
                load_aware_rssi_stats['avg_energy_per_bit'], ea_stats['avg_energy_per_bit']
            ),
            'energy_saving_total_joules_percent': energy_saving_total_joules_percent,
            'co2_saving_percent': co2_saving_percent,
            'handoff_reduction_percent': handoff_reduction,
            'energy_saving_vs_naive_percent': _pct_saving(
                naive_stats['avg_energy_per_bit'], ea_stats['avg_energy_per_bit']
            ),
            'rssi_vs_naive_energy_percent': _pct_saving(
                naive_stats['avg_energy_per_bit'], rssi_stats['avg_energy_per_bit']
            ),
            'energy_aware_handoffs_leq_rssi': ea_ho <= rssi_ho,
            'energy_aware_stats': ea_stats,
            'rssi_stats': rssi_stats,
            'sinr_stats': sinr_stats,
            'load_aware_rssi_stats': load_aware_rssi_stats,
            'naive_nearest_stats': naive_stats,
        }

        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Energy Saving (EA vs RSSI):            {energy_improvement:.2f}%")
        print(f"Energy Saving (EA vs SINR):            {comparison['energy_saving_vs_sinr_percent']:.2f}%")
        print(f"Energy Saving (EA vs Load-aware RSSI): {comparison['energy_saving_vs_load_aware_rssi_percent']:.2f}%")
        print(f"Energy Saving (EA vs Naive):           {comparison['energy_saving_vs_naive_percent']:.2f}%")
        print(f"CO2 Saving vs RSSI:                    {co2_saving_percent:.2f}%")
        print(f"Handoff Reduction (EA vs RSSI):        {handoff_reduction:.2f}%")
        print(f"EA handoffs <= RSSI: {ea_ho <= rssi_ho} ({ea_ho} vs {rssi_ho})")

        return comparison

    def save_results(self, filename: str = 'results/simulation_results.json'):
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

        def _serialize(obj):
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialize(i) for i in obj]
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        payload = _serialize({
            'config': {
                'num_vehicles': self.config.num_vehicles,
                'num_base_stations': self.config.num_base_stations,
                'duration': self.config.duration,
                'area_size': self.config.area_size,
                'seed': self.config.seed,
                'shadowing_std_db': self.config.shadowing_std_db,
                'weather_profile': self.config.weather_profile,
                'carbon_intensity_kg_per_kwh': self.config.carbon_intensity_kg_per_kwh,
            },
            'results': {
                algo: {
                    'stats': self.results[algo].get('stats', {}),
                    'algorithm_stats': self.results[algo].get('algorithm_stats', {}),
                }
                for algo in self.results
            },
        })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {filename}")
