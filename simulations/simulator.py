import json
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from src.models.vehicle import Vehicle
from src.models.basestation import BaseStation, BSConfig
from src.models.energy import EnergyModel, EnvironmentalMetrics
from src.algorithms.energy_aware_handoff import EnergyAwareHandoff
from src.algorithms.rssi_handoff import RSSIHandoff
from src.algorithms.naive_nearest_handoff import NaiveNearestHandoff
from simulations.config import SimulationConfig


class V2XSimulator:
    """
    Complete V2X Network Simulator
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        np.random.seed(self.config.seed)

        self.base_stations: List[BaseStation] = []
        self.vehicles: List[Vehicle] = []
        self.energy_model = EnergyModel()
        self.environmental = EnvironmentalMetrics(
            self.config.carbon_intensity_kg_per_kwh
        )

        self.energy_aware_algo = EnergyAwareHandoff(
            snr_outage_threshold_db=self.config.snr_outage_threshold_db,
            hysteresis=self.config.energy_aware_min_energy_saving,
            time_to_trigger_s=self.config.energy_aware_time_to_trigger_s,
            min_time_since_last_handoff_s=self.config.handoff_cooldown_s,
            min_data_rate_bps=self.config.energy_aware_min_data_rate_bps,
        )
        self.rssi_algo = RSSIHandoff()
        self.naive_nearest_algo = NaiveNearestHandoff()

        self.results = {
            'energy_aware': {},
            'rssi': {},
            'naive_nearest': {},
        }

    def setup_network(self):
        """Create base station grid topology"""
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
                )

                bs = BaseStation(bs_id=bs_id, x=x, y=y, config=bs_config)
                self.base_stations.append(bs)
                bs_id += 1

        print(f"Created {len(self.base_stations)} base stations")

    def setup_vehicles(self):
        """Create vehicles with random positions and speeds"""
        self.vehicles = []

        for i in range(self.config.num_vehicles):
            speed = np.random.uniform(
                self.config.vehicle_speed_min,
                self.config.vehicle_speed_max
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

    def _execute_handoff(self, algo, vehicle: Vehicle, old_bs, new_bs, current_time: float):
        algo.execute_handoff(vehicle, old_bs, new_bs)
        vehicle.state.energy_consumed += self.config.handoff_energy_joules
        distance = new_bs.distance_to(vehicle.x, vehicle.y)
        tx_power = self.energy_model.calculate_tx_power_required(distance)
        vehicle.state.energy_consumed += tx_power * self.config.handoff_delay_s
        vehicle.state.last_handoff_time = current_time

    def _maybe_highway_lane_switch(self, vehicle: Vehicle, current_time: float):
        """Rare adjacent-lane change; resamples speed for the new lane."""
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

    def run_algorithm(self, algorithm_name: str) -> Dict:
        print(f"\n{'='*60}")
        print(f"Running {algorithm_name.upper()} handoff algorithm")
        print(f"{'='*60}")

        for v in self.vehicles:
            v.reset_stats()
        for bs in self.base_stations:
            bs.connected_vehicles = []

        if algorithm_name == 'energy_aware':
            algo = self.energy_aware_algo
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
            'avg_distance': []
        }

        for vehicle in self.vehicles:
            if algorithm_name == 'energy_aware':
                best_bs, info = algo.select_best_bs(vehicle, self.base_stations)
            elif algorithm_name == 'naive_nearest':
                best_bs, info = algo.select_best_bs(vehicle, self.base_stations)
            else:
                best_bs, info = algo.select_best_bs(
                    vehicle, self.base_stations, self.config.tx_power_default
                )

            if best_bs:
                self._execute_handoff(algo, vehicle, None, best_bs, 0.0)

        num_steps = int(self.config.duration / self.config.time_step)

        for step in tqdm(range(num_steps), desc=f"Simulating {algorithm_name}"):
            current_time = step * self.config.time_step

            step_energy = 0
            step_epb = 0
            step_tx_power = 0
            step_data_rate = 0
            step_distance = 0
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
                        None
                    )

                if algorithm_name == 'energy_aware':
                    candidate_bs, info = algo.select_best_bs(vehicle, self.base_stations)
                    if candidate_bs:
                        candidate_epb = info['energy_per_bit']
                elif algorithm_name == 'naive_nearest':
                    candidate_bs, info = algo.select_best_bs(
                        vehicle, self.base_stations
                    )
                    if candidate_bs:
                        cand_epb = self.energy_aware_algo.link_energy_per_bit(
                            vehicle, candidate_bs
                        )
                        candidate_epb = cand_epb if cand_epb is not None else float(
                            "inf"
                        )
                else:
                    candidate_bs, info = algo.select_best_bs(
                        vehicle, self.base_stations, self.config.tx_power_default
                    )
                    if candidate_bs:
                        cand_epb = self.energy_aware_algo.link_energy_per_bit(
                            vehicle, candidate_bs
                        )
                        candidate_epb = cand_epb if cand_epb is not None else float("inf")

                if candidate_bs is None:
                    continue

                should_ho = False
                if current_bs is None:
                    should_ho = True
                elif algorithm_name == 'energy_aware':
                    current_epb = algo.link_energy_per_bit(vehicle, current_bs)
                    if current_epb is None:
                        current_epb = float("inf")
                    should_ho = algo.should_handoff(
                        vehicle,
                        current_bs,
                        candidate_bs,
                        current_epb,
                        candidate_epb,
                        current_time=current_time,
                    )
                elif algorithm_name == 'naive_nearest':
                    should_ho = algo.should_handoff(
                        vehicle, current_bs, candidate_bs
                    )
                else:
                    current_d = current_bs.distance_to(vehicle.x, vehicle.y)
                    cand_d = candidate_bs.distance_to(vehicle.x, vehicle.y)
                    current_tx = self.energy_model.calculate_tx_power_required(current_d)
                    cand_tx = self.energy_model.calculate_tx_power_required(cand_d)
                    current_rssi = current_bs.calculate_received_power(
                        vehicle.x, vehicle.y, current_tx
                    )
                    candidate_rssi = candidate_bs.calculate_received_power(
                        vehicle.x, vehicle.y, cand_tx
                    )
                    should_ho = algo.should_handoff(current_rssi, candidate_rssi)

                cooldown_ok = (
                    current_time - vehicle.state.last_handoff_time
                    >= self.config.handoff_cooldown_s
                )
                if should_ho and current_bs != candidate_bs and cooldown_ok:
                    self._execute_handoff(algo, vehicle, current_bs, candidate_bs, current_time)
                    current_bs = candidate_bs

                if vehicle.state.connected_bs_id is not None:
                    distance = current_bs.distance_to(vehicle.x, vehicle.y)
                    if algorithm_name == 'rssi' and self.config.rssi_energy_use_fixed_tx:
                        tx_power = self.config.tx_power_default
                        data_rate = self.config.data_rate
                    else:
                        tx_power = self.energy_model.calculate_tx_power_required(distance)
                        data_rate = self.config.data_rate
                        tdr = self.energy_aware_algo.link_tx_and_data_rate(
                            vehicle, current_bs
                        )
                        if tdr is not None:
                            tx_power, data_rate = tdr

                    vehicle.update_energy(
                        tx_power=tx_power,
                        duration=self.config.time_step,
                        data_rate=data_rate,
                        energy_model=self.energy_model,
                    )

                    step_energy += self.energy_model.calculate_total_power(
                        tx_power, 'transmit'
                    ) * self.config.time_step
                    step_epb += self.energy_model.calculate_energy_per_bit(
                        tx_power, data_rate
                    )
                    step_tx_power += tx_power
                    step_data_rate += data_rate
                    step_distance += distance
                    connected_count += 1

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

        total_energy = sum(v.state.energy_consumed for v in self.vehicles)
        total_bits = sum(v.state.bits_transmitted for v in self.vehicles)
        total_handoffs = algo.total_handoffs

        co2_kg = self.environmental.energy_to_co2(total_energy)
        dur_s = float(self.config.duration)
        sy = self.config.seconds_per_year
        co2_avg_v = self.environmental.avg_co2_kg_per_vehicle(
            co2_kg, self.config.num_vehicles
        )
        co2_v_yr = self.environmental.co2_kg_per_vehicle_per_year(
            co2_kg, self.config.num_vehicles, dur_s, sy
        )
        stats = {
            'total_energy_joules': total_energy,
            'total_bits': total_bits,
            'avg_energy_per_bit': total_energy / total_bits if total_bits > 0 else 0,
            'total_handoffs': total_handoffs,
            'avg_handoffs_per_vehicle': total_handoffs / self.config.num_vehicles,
            'avg_tx_power': float(np.mean(metrics['avg_tx_power'])),
            'avg_data_rate': float(np.mean(metrics['avg_data_rate'])),
            'connection_rate': np.mean(metrics['connected_vehicles']) / self.config.num_vehicles,
            'co2_kg': co2_kg,
            'co2_grams': co2_kg * 1000,
            'carbon_intensity_kg_per_kwh': self.environmental.carbon_intensity,
            'avg_co2_kg_per_vehicle': co2_avg_v,
            'co2_kg_per_vehicle_per_year': co2_v_yr,
            'simulation_duration_s': dur_s,
            'seconds_per_year': sy,
            'per_vehicle_energy_joules': [
                float(v.state.energy_consumed) for v in self.vehicles
            ],
        }

        print(f"\n{algorithm_name.upper()} Results:")
        print(f"  Total Energy: {total_energy:.2f} J")
        print(
            f"  Est. CO2 (intensity {self.environmental.carbon_intensity:g} kg/kWh): "
            f"{co2_kg:.6f} kg ({co2_kg * 1000:.3f} g)"
        )
        print(
            f"  Avg CO2 / vehicle (sim): {co2_avg_v:.6f} kg; "
            f"extrapolated / vehicle / year: {co2_v_yr:.6f} kg"
        )
        print(f"  Energy-per-Bit: {stats['avg_energy_per_bit']*1e6:.4f} uJ/bit")
        print(f"  Total Handoffs: {total_handoffs}")
        print(f"  Avg TX Power: {stats['avg_tx_power']*1000:.2f} mW")

        return {
            'metrics': metrics,
            'stats': stats,
            'algorithm_stats': algo.get_statistics()
        }

    def run_comparison(self):
        self.setup_network()
        self.setup_vehicles()

        self.results['energy_aware'] = self.run_algorithm('energy_aware')

        np.random.seed(self.config.seed)
        self.setup_vehicles()

        self.results['rssi'] = self.run_algorithm('rssi')

        np.random.seed(self.config.seed)
        self.setup_vehicles()

        self.results['naive_nearest'] = self.run_algorithm('naive_nearest')

        ea_stats = self.results['energy_aware']['stats']
        rssi_stats = self.results['rssi']['stats']
        naive_stats = self.results['naive_nearest']['stats']

        energy_improvement = (
            (rssi_stats['avg_energy_per_bit'] - ea_stats['avg_energy_per_bit']) /
            rssi_stats['avg_energy_per_bit'] * 100
        ) if rssi_stats['avg_energy_per_bit'] > 0 else 0.0

        handoff_reduction = (
            (rssi_stats['total_handoffs'] - ea_stats['total_handoffs']) /
            max(1, rssi_stats['total_handoffs']) * 100
        )

        energy_saving_vs_naive = (
            (naive_stats['avg_energy_per_bit'] - ea_stats['avg_energy_per_bit']) /
            naive_stats['avg_energy_per_bit'] * 100
        ) if naive_stats['avg_energy_per_bit'] > 0 else 0.0

        if self.config.rssi_energy_use_fixed_tx:
            rssi_vs_naive_energy = None
        else:
            rssi_vs_naive_energy = (
                (naive_stats['avg_energy_per_bit'] - rssi_stats['avg_energy_per_bit']) /
                naive_stats['avg_energy_per_bit'] * 100
            ) if naive_stats['avg_energy_per_bit'] > 0 else 0.0

        rj = rssi_stats['total_energy_joules']
        ej = ea_stats['total_energy_joules']
        energy_saving_total_joules_percent = (
            (rj - ej) / rj * 100.0 if rj > 0 else 0.0
        )

        rc = rssi_stats['co2_kg']
        ec = ea_stats['co2_kg']
        co2_saving_percent = (
            (rc - ec) / rc * 100.0 if rc > 0 else 0.0
        )

        ea_ho = ea_stats['total_handoffs']
        rssi_ho = rssi_stats['total_handoffs']
        energy_aware_handoffs_leq_rssi = ea_ho <= rssi_ho

        comparison = {
            'energy_saving_percent': energy_improvement,
            'energy_saving_total_joules_percent': energy_saving_total_joules_percent,
            'co2_saving_percent': co2_saving_percent,
            'handoff_reduction_percent': handoff_reduction,
            'energy_saving_vs_naive_percent': energy_saving_vs_naive,
            'rssi_vs_naive_energy_percent': rssi_vs_naive_energy,
            'energy_aware_handoffs_leq_rssi': energy_aware_handoffs_leq_rssi,
            'energy_aware_stats': ea_stats,
            'rssi_stats': rssi_stats,
            'naive_nearest_stats': naive_stats,
        }

        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Energy Saving (vs RSSI): {energy_improvement:.2f}%")
        print(
            f"Energy Saving vs RSSI (total joules): {energy_saving_total_joules_percent:.2f}%"
        )
        print(f"CO2 Saving vs RSSI (reporting): {co2_saving_percent:.2f}%")
        print(f"Handoff Reduction (vs RSSI): {handoff_reduction:.2f}%")
        print(
            f"Energy-Aware handoffs <= RSSI baseline: {energy_aware_handoffs_leq_rssi} "
            f"({ea_ho} vs {rssi_ho})"
        )
        print(f"Energy Saving (vs Naive Nearest): {energy_saving_vs_naive:.2f}%")
        if rssi_vs_naive_energy is None:
            print("RSSI vs Naive (EPB): N/A (RSSI uses fixed-TX energy model)")
        else:
            print(f"RSSI vs Naive (energy improvement): {rssi_vs_naive_energy:.2f}%")

        return comparison

    def save_results(self, filename: str = 'results/simulation_results.json'):
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_results = convert_to_serializable({
            'config': {
                'num_vehicles': self.config.num_vehicles,
                'num_base_stations': self.config.num_base_stations,
                'duration': self.config.duration,
                'area_size': self.config.area_size,
                'seed': self.config.seed,
                'shadowing_std_db': self.config.shadowing_std_db,
                'weather_profile': self.config.weather_profile,
                'weather_path_loss_exponent': self.config.get_weather().path_loss_exponent,
                'weather_shadowing_std_db': self.config.get_weather().shadowing_std_db,
                'weather_rain_attenuation_db_per_km': self.config.get_weather().rain_attenuation_db_per_km,
                'highway_lateral_noise_std_m': self.config.highway_lateral_noise_std_m,
                'carbon_intensity_kg_per_kwh': self.config.carbon_intensity_kg_per_kwh,
                'seconds_per_year': self.config.seconds_per_year,
                'handoff_cooldown_s': self.config.handoff_cooldown_s,
                'energy_aware_min_energy_saving': self.config.energy_aware_min_energy_saving,
                'energy_aware_time_to_trigger_s': self.config.energy_aware_time_to_trigger_s,
                'energy_aware_min_data_rate_bps': self.config.energy_aware_min_data_rate_bps,
                'rssi_energy_use_fixed_tx': self.config.rssi_energy_use_fixed_tx,
            },
            'results': self.results
        })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {filename}")
