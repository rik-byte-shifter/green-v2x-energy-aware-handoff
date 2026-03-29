import json
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from src.models.vehicle import Vehicle
from src.models.basestation import BaseStation, BSConfig
from src.models.energy import EnergyModel
from src.algorithms.energy_aware_handoff import EnergyAwareHandoff
from src.algorithms.rssi_handoff import RSSIHandoff
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

        self.energy_aware_algo = EnergyAwareHandoff()
        self.rssi_algo = RSSIHandoff()

        self.results = {
            'energy_aware': {},
            'rssi': {}
        }

    def setup_network(self):
        """Create base station grid topology"""
        self.base_stations = []

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
                    coverage_radius=self.config.bs_coverage_radius
                )

                bs = BaseStation(bs_id=bs_id, x=x, y=y, config=bs_config)
                self.base_stations.append(bs)
                bs_id += 1

        print(f"Created {len(self.base_stations)} base stations")

    def setup_vehicles(self):
        """Create vehicles with random positions and speeds"""
        self.vehicles = []

        for i in range(self.config.num_vehicles):
            x = np.random.uniform(0, self.config.area_size)
            y = np.random.uniform(0, self.config.area_size)
            speed = np.random.uniform(
                self.config.vehicle_speed_min,
                self.config.vehicle_speed_max
            )
            direction = np.random.uniform(0, 2 * np.pi)

            vehicle = Vehicle(
                vehicle_id=i,
                x=x,
                y=y,
                speed=speed,
                direction=direction
            )
            self.vehicles.append(vehicle)

        print(f"Created {len(self.vehicles)} vehicles")

    def _execute_handoff(self, algo, vehicle: Vehicle, old_bs, new_bs, current_time: float):
        algo.execute_handoff(vehicle, old_bs, new_bs)
        vehicle.state.energy_consumed += self.config.handoff_energy_joules
        vehicle.state.last_handoff_time = current_time

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

            for vehicle in self.vehicles:
                vehicle.move(
                    delta_time=self.config.time_step,
                    boundary=(0, self.config.area_size)
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
                else:
                    candidate_bs, info = algo.select_best_bs(
                        vehicle, self.base_stations, self.config.tx_power_default
                    )
                    if candidate_bs:
                        distance = candidate_bs.distance_to(vehicle.x, vehicle.y)
                        candidate_tx_power = self.energy_model.calculate_tx_power_required(distance)
                        candidate_epb = self.energy_model.calculate_energy_per_bit(
                            candidate_tx_power, self.config.data_rate
                        )

                if candidate_bs is None:
                    continue

                should_ho = False
                if current_bs is None:
                    should_ho = True
                elif algorithm_name == 'energy_aware':
                    current_distance = current_bs.distance_to(vehicle.x, vehicle.y)
                    current_tx_power = self.energy_model.calculate_tx_power_required(current_distance)
                    current_epb = self.energy_model.calculate_energy_per_bit(
                        current_tx_power, self.config.data_rate
                    )
                    should_ho = algo.should_handoff(
                        vehicle, current_bs, candidate_bs, current_epb, candidate_epb
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
                    tx_power = self.energy_model.calculate_tx_power_required(distance)

                    vehicle.update_energy(
                        tx_power=tx_power,
                        duration=self.config.time_step,
                        data_rate=self.config.data_rate,
                        energy_model=self.energy_model,
                    )

                    step_energy += self.energy_model.calculate_total_power(
                        tx_power, 'transmit'
                    ) * self.config.time_step
                    step_epb += self.energy_model.calculate_energy_per_bit(
                        tx_power, self.config.data_rate
                    )
                    step_tx_power += tx_power
                    step_data_rate += self.config.data_rate
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

        stats = {
            'total_energy_joules': total_energy,
            'total_bits': total_bits,
            'avg_energy_per_bit': total_energy / total_bits if total_bits > 0 else 0,
            'total_handoffs': total_handoffs,
            'avg_handoffs_per_vehicle': total_handoffs / self.config.num_vehicles,
            'avg_tx_power': float(np.mean(metrics['avg_tx_power'])),
            'avg_data_rate': float(np.mean(metrics['avg_data_rate'])),
            'connection_rate': np.mean(metrics['connected_vehicles']) / self.config.num_vehicles
        }

        print(f"\n{algorithm_name.upper()} Results:")
        print(f"  Total Energy: {total_energy:.2f} J")
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

        ea_stats = self.results['energy_aware']['stats']
        rssi_stats = self.results['rssi']['stats']

        energy_improvement = (
            (rssi_stats['avg_energy_per_bit'] - ea_stats['avg_energy_per_bit']) /
            rssi_stats['avg_energy_per_bit'] * 100
        ) if rssi_stats['avg_energy_per_bit'] > 0 else 0.0

        handoff_reduction = (
            (rssi_stats['total_handoffs'] - ea_stats['total_handoffs']) /
            max(1, rssi_stats['total_handoffs']) * 100
        )

        comparison = {
            'energy_saving_percent': energy_improvement,
            'handoff_reduction_percent': handoff_reduction,
            'energy_aware_stats': ea_stats,
            'rssi_stats': rssi_stats
        }

        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Energy Saving: {energy_improvement:.2f}%")
        print(f"Handoff Reduction: {handoff_reduction:.2f}%")

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
                'area_size': self.config.area_size
            },
            'results': self.results
        })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {filename}")
