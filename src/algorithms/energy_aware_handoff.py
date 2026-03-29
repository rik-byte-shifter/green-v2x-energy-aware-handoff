import numpy as np
from typing import List, Tuple, Optional
from src.models.vehicle import Vehicle
from src.models.basestation import BaseStation
from src.models.energy import EnergyModel


class EnergyAwareHandoff:
    """
    Energy-Aware Handoff Algorithm

    Selects base station based on:
    1. Minimum energy-per-bit
    2. BS load balancing
    3. Hysteresis to prevent ping-pong
    """

    def __init__(self, hysteresis: float = 0.35,
                 load_threshold: float = 0.98):
        self.hysteresis = hysteresis
        self.load_threshold = load_threshold
        self.energy_model = EnergyModel()

        self.total_handoffs = 0
        self.handoff_history = []

    def reset_statistics(self):
        self.total_handoffs = 0
        self.handoff_history = []

    def select_best_bs(self, vehicle: Vehicle,
                      base_stations: List[BaseStation]) -> Tuple[Optional[BaseStation], dict]:
        candidates = []

        for bs in base_stations:
            if not bs.is_in_coverage(vehicle.x, vehicle.y):
                continue

            if not bs.has_capacity():
                continue

            distance = bs.distance_to(vehicle.x, vehicle.y)

            tx_power = self.energy_model.calculate_tx_power_required(distance)

            rx_power = bs.calculate_received_power(vehicle.x, vehicle.y, tx_power)
            snr = rx_power - (-174 + 10 * np.log10(20e6))
            data_rate = 20e6 * np.log2(1 + 10 ** (snr / 10)) if snr > 0 else 0

            epb = self.energy_model.calculate_energy_per_bit(
                tx_power, data_rate
            )

            load_penalty = 1 + bs.get_load()

            metric = epb * load_penalty

            candidates.append({
                'bs': bs,
                'distance': distance,
                'tx_power': tx_power,
                'data_rate': data_rate,
                'energy_per_bit': epb,
                'metric': metric,
                'load': bs.get_load()
            })

        if not candidates:
            return None, {}

        best = min(candidates, key=lambda x: x['metric'])

        return best['bs'], {
            'tx_power': best['tx_power'],
            'energy_per_bit': best['energy_per_bit'],
            'data_rate': best['data_rate'],
            'distance': best['distance']
        }

    def should_handoff(self, vehicle: Vehicle,
                      current_bs: BaseStation,
                      candidate_bs: BaseStation,
                      current_epb: float,
                      candidate_epb: float) -> bool:
        if current_bs.get_load() > self.load_threshold:
            return True

        if current_epb == 0:
            return True

        energy_saving = (current_epb - candidate_epb) / current_epb

        return energy_saving > self.hysteresis

    def execute_handoff(self, vehicle: Vehicle,
                       old_bs: Optional[BaseStation],
                       new_bs: BaseStation):
        if old_bs is not None:
            old_bs.remove_vehicle(vehicle.vehicle_id)

        new_bs.add_vehicle(vehicle.vehicle_id)
        vehicle.state.connected_bs_id = new_bs.bs_id
        vehicle.state.handoff_count += 1
        self.total_handoffs += 1

        self.handoff_history.append({
            'vehicle_id': vehicle.vehicle_id,
            'from_bs': old_bs.bs_id if old_bs else None,
            'to_bs': new_bs.bs_id,
            'timestamp': len(self.handoff_history)
        })

    def get_statistics(self) -> dict:
        n_vehicles = len(set(h['vehicle_id'] for h in self.handoff_history))
        return {
            'total_handoffs': self.total_handoffs,
            'avg_handoffs_per_vehicle': self.total_handoffs / max(1, n_vehicles)
        }
