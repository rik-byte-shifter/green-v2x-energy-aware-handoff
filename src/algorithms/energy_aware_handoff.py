import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from src.models.vehicle import Vehicle
from src.models.basestation import BaseStation
from src.models.energy import EnergyModel

_NOISE_PLUS_TERM = -174.0 + 10 * np.log10(20e6)


class EnergyAwareHandoff:
    """
    Energy-Aware Handoff Algorithm

    Selects base station based on:
    1. Minimum energy-per-bit
    2. BS load balancing
    3. QoS: optional minimum achievable data rate (bps) on the link
    4. Hysteresis + Time-to-Trigger (TTT) to limit ping-pong
    """

    def __init__(
        self,
        hysteresis: float = 0.25,
        load_threshold: float = 0.98,
        snr_outage_threshold_db: float = 0.0,
        time_to_trigger_s: float = 2.0,
        min_time_since_last_handoff_s: float = 4.0,
        min_data_rate_bps: float = 5e6,
    ):
        # Minimum relative EPB improvement required (e.g. 0.25 => >25% vs current BS)
        self.hysteresis = hysteresis
        self.load_threshold = load_threshold
        self.snr_outage_threshold_db = snr_outage_threshold_db
        self.time_to_trigger_s = time_to_trigger_s
        self.min_time_since_last_handoff_s = min_time_since_last_handoff_s
        self.min_data_rate_bps = min_data_rate_bps
        self.energy_model = EnergyModel()

        self.total_handoffs = 0
        self.handoff_history = []

        # TTT: per vehicle, which candidate BS id we are timing and when we started
        self._ttt_candidate_bs_id: Dict[int, int] = {}
        self._ttt_since: Dict[int, float] = {}

    def reset_statistics(self):
        self.total_handoffs = 0
        self.handoff_history = []
        self._ttt_candidate_bs_id.clear()
        self._ttt_since.clear()

    def _clear_ttt(self, vehicle_id: int) -> None:
        self._ttt_candidate_bs_id.pop(vehicle_id, None)
        self._ttt_since.pop(vehicle_id, None)

    def _link_metrics(
        self,
        vehicle: Vehicle,
        bs: BaseStation,
        *,
        require_capacity: bool,
    ) -> Optional[Dict[str, Any]]:
        if not bs.is_in_coverage(vehicle.x, vehicle.y):
            return None
        if require_capacity and not bs.has_capacity():
            return None

        distance = bs.distance_to(vehicle.x, vehicle.y)
        tx_power = self.energy_model.calculate_tx_power_required(distance)
        rx_power = bs.calculate_received_power(vehicle.x, vehicle.y, tx_power)
        snr = rx_power - _NOISE_PLUS_TERM
        data_rate = max(
            1e5,
            20e6 * np.log2(1.0 + 10 ** (snr / 10.0)),
        )
        epb = self.energy_model.calculate_energy_per_bit(tx_power, data_rate)
        load_factor = 1.0 + 2.0 * bs.get_load()
        metric = epb * load_factor

        return {
            'bs': bs,
            'distance': distance,
            'tx_power': tx_power,
            'data_rate': data_rate,
            'energy_per_bit': epb,
            'metric': metric,
            'load': bs.get_load(),
        }

    def link_energy_per_bit(self, vehicle: Vehicle, bs: BaseStation) -> Optional[float]:
        """EPB for an existing link (ignores capacity gate)."""
        m = self._link_metrics(vehicle, bs, require_capacity=False)
        return None if m is None else m['energy_per_bit']

    def link_tx_and_data_rate(
        self, vehicle: Vehicle, bs: BaseStation
    ) -> Optional[Tuple[float, float]]:
        """TX power and Shannon-style data rate for the vehicle–BS link."""
        m = self._link_metrics(vehicle, bs, require_capacity=False)
        if m is None:
            return None
        return m['tx_power'], m['data_rate']

    def select_best_bs(self, vehicle: Vehicle,
                      base_stations: List[BaseStation]) -> Tuple[Optional[BaseStation], dict]:
        candidates = []
        for bs in base_stations:
            row = self._link_metrics(vehicle, bs, require_capacity=True)
            if row is not None:
                candidates.append(row)

        if not candidates:
            return None, {}

        qos_pool = [
            r for r in candidates
            if r['data_rate'] >= self.min_data_rate_bps
        ]
        # Prefer BS that meet min throughput; if none (outage / edge), fall back to best EPB
        pool = qos_pool if qos_pool else candidates
        qos_satisfied = bool(qos_pool)

        best = min(pool, key=lambda x: x['metric'])

        return best['bs'], {
            'tx_power': best['tx_power'],
            'energy_per_bit': best['energy_per_bit'],
            'data_rate': best['data_rate'],
            'distance': best['distance'],
            'metric': best['metric'],
            'qos_met': qos_satisfied,
            'min_data_rate_bps': self.min_data_rate_bps,
        }

    def should_handoff(
        self,
        vehicle: Vehicle,
        current_bs: BaseStation,
        candidate_bs: BaseStation,
        current_epb: float,
        candidate_epb: float,
        current_time: Optional[float] = None,
    ) -> bool:
        vid = vehicle.vehicle_id

        if (
            current_bs.get_load() > 0.8
            and candidate_bs.get_load() < 0.6
        ):
            self._clear_ttt(vid)
            return True

        if current_bs.get_load() > self.load_threshold:
            self._clear_ttt(vid)
            return True

        if current_epb == 0 or np.isinf(current_epb):
            self._clear_ttt(vid)
            return True

        time_since_last_handoff = (
            float('inf')
            if current_time is None
            else (current_time - vehicle.state.last_handoff_time)
        )
        if time_since_last_handoff < self.min_time_since_last_handoff_s:
            self._clear_ttt(vid)
            return False

        if current_epb <= 0:
            self._clear_ttt(vid)
            return False

        energy_saving = (current_epb - candidate_epb) / current_epb
        energy_wants = energy_saving > self.hysteresis

        if not energy_wants:
            self._clear_ttt(vid)
            return False

        # Time-to-Trigger: candidate must beat threshold and stay best for TTT
        if current_time is None or self.time_to_trigger_s <= 0.0:
            self._clear_ttt(vid)
            return True

        cid = candidate_bs.bs_id
        if self._ttt_candidate_bs_id.get(vid) != cid:
            self._ttt_candidate_bs_id[vid] = cid
            self._ttt_since[vid] = current_time
            return False

        if current_time - self._ttt_since[vid] >= self.time_to_trigger_s:
            self._clear_ttt(vid)
            return True

        return False

    def execute_handoff(self, vehicle: Vehicle,
                       old_bs: Optional[BaseStation],
                       new_bs: BaseStation):
        self._clear_ttt(vehicle.vehicle_id)

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
