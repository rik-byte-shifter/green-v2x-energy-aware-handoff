import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
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
        packet_size: int = 1000,
        sinr_handoff_trigger_db: float = 3.0,
        max_acceptable_epb: float = 2.0e-7,
        connectivity_penalty_factor: float = 2.0,
    ):
        # Minimum relative EPB improvement required (e.g. 0.25 => >25% vs current BS)
        self.hysteresis = hysteresis
        self.load_threshold = load_threshold
        self.snr_outage_threshold_db = snr_outage_threshold_db
        self.time_to_trigger_s = time_to_trigger_s
        self.min_time_since_last_handoff_s = min_time_since_last_handoff_s
        self.min_data_rate_bps = min_data_rate_bps
        self.packet_size = packet_size
        self.sinr_handoff_trigger_db = sinr_handoff_trigger_db
        self.max_acceptable_epb = max_acceptable_epb
        self.connectivity_penalty_factor = connectivity_penalty_factor
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
        link_metrics_getter: Optional[
            Callable[[Vehicle, BaseStation, bool], Optional[Dict[str, Any]]]
        ] = None,
    ) -> Optional[Dict[str, Any]]:
        if link_metrics_getter is not None:
            row = link_metrics_getter(vehicle, bs, require_capacity)
            if row is None:
                return None
            # Enforce outage policy even when metrics are provided by simulator cache.
            # This keeps candidate filtering consistent with the configured threshold.
            if float(row.get('snr', float('inf'))) <= self.snr_outage_threshold_db:
                out = dict(row)
                out['data_rate'] = 0.0
                out['energy_per_bit'] = float('inf')
                out['metric'] = float('inf')
                return out
            return row

        if not bs.is_in_coverage(vehicle.x, vehicle.y):
            return None
        if require_capacity and not bs.has_capacity():
            return None

        distance = bs.distance_to(vehicle.x, vehicle.y)
        # Weather-aware TX requirement: must use the same channel/path-loss
        # model as `calculate_received_power()` / SNR.
        tx_power = bs.calculate_tx_power_required_for_target_rx(distance)
        rx_power = bs.calculate_received_power(vehicle.x, vehicle.y, tx_power)
        snr = rx_power - _NOISE_PLUS_TERM
        if snr <= self.snr_outage_threshold_db:
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
                packet_size=self.packet_size,
                device_type="obu",
            )
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
            'snr': snr,
        }

    def link_energy_per_bit(
        self,
        vehicle: Vehicle,
        bs: BaseStation,
        *,
        link_metrics_getter: Optional[
            Callable[[Vehicle, BaseStation, bool], Optional[Dict[str, Any]]]
        ] = None,
    ) -> Optional[float]:
        """EPB for an existing link (ignores capacity gate)."""
        m = self._link_metrics(
            vehicle,
            bs,
            require_capacity=False,
            link_metrics_getter=link_metrics_getter,
        )
        return None if m is None else m['energy_per_bit']

    def link_tx_and_data_rate(
        self,
        vehicle: Vehicle,
        bs: BaseStation,
        *,
        link_metrics_getter: Optional[
            Callable[[Vehicle, BaseStation, bool], Optional[Dict[str, Any]]]
        ] = None,
    ) -> Optional[Tuple[float, float]]:
        """TX power and Shannon-style data rate for the vehicle–BS link."""
        m = self._link_metrics(
            vehicle,
            bs,
            require_capacity=False,
            link_metrics_getter=link_metrics_getter,
        )
        if m is None:
            return None
        return m['tx_power'], m['data_rate']

    def select_best_bs(self, vehicle: Vehicle,
                      base_stations: List[BaseStation],
                      *,
                      link_metrics_getter: Optional[
                          Callable[[Vehicle, BaseStation, bool], Optional[Dict[str, Any]]]
                      ] = None) -> Tuple[Optional[BaseStation], dict]:
        candidates = []
        for bs in base_stations:
            row = self._link_metrics(
                vehicle,
                bs,
                require_capacity=True,
                link_metrics_getter=link_metrics_getter,
            )
            if row is not None:
                candidates.append(row)

        if not candidates:
            return None, {}

        # Explicitly remove outage links (zero throughput) from handoff candidates.
        viable = [r for r in candidates if r['data_rate'] > 0.0]
        if not viable:
            return None, {}

        # 🔽 HARD SINR + QoS FILTER
        MIN_SINR_FOR_SELECTION = -2.0  # dB: below this, link is effectively dead
        qos_pool = [
            r for r in viable
            if r['data_rate'] >= self.min_data_rate_bps
            and r.get('snr', -999) > MIN_SINR_FOR_SELECTION
        ]

        # Fallback strategy: prefer QoS-compliant, but never pick a dead link
        if qos_pool:
            pool = qos_pool
        else:
            pool = [r for r in viable if r.get('snr', -999) > MIN_SINR_FOR_SELECTION]
            if not pool:
                pool = viable  # Absolute last resort: maintain connectivity

        # 🔽 CELL-EDGE PENALTY
        for r in pool:
            distance_ratio = r['distance'] / r['bs'].config.coverage_radius
            if distance_ratio > 0.75:  # Near cell edge → penalize metric
                r['adjusted_metric'] = r['metric'] * self.connectivity_penalty_factor
            else:
                r['adjusted_metric'] = r['metric']

        best = min(pool, key=lambda x: x['adjusted_metric'])
        return best['bs'], {
            'tx_power': best['tx_power'],
            'energy_per_bit': best['energy_per_bit'],
            'data_rate': best['data_rate'],
            'distance': best['distance'],
            'metric': best['adjusted_metric'],
            'qos_met': bool(qos_pool),
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
        current_sinr_db: Optional[float] = None,
        candidate_sinr_db: Optional[float] = None,
    ) -> bool:
        vid = vehicle.vehicle_id

        # 0) Hard cooldown guard first.
        # Prevents "forced" triggers from bypassing anti-ping-pong protection.
        time_since_last = (
            float('inf')
            if current_time is None
            else current_time - vehicle.state.last_handoff_time
        )
        if time_since_last < self.min_time_since_last_handoff_s:
            return False

        forced_handoff = False

        # 1️⃣ Overload escape
        if current_bs.get_load() > self.load_threshold:
            forced_handoff = True

        # 2️⃣ SINR degradation trigger.
        # Only treat as emergency when the candidate is clearly better;
        # otherwise this trigger can create oscillations near cell edges.
        if (
            not forced_handoff
            and current_sinr_db is not None
            and candidate_sinr_db is not None
            and current_sinr_db < self.sinr_handoff_trigger_db
            and candidate_sinr_db >= current_sinr_db + 1.0
            and candidate_sinr_db > self.snr_outage_threshold_db + 1.0
        ):
            forced_handoff = True

        # 3️⃣ Max EPB trigger (prevent energy waste on poor links)
        if (
            not forced_handoff
            and current_epb > self.max_acceptable_epb
            and candidate_epb < current_epb * 0.95
        ):
            forced_handoff = True

        # 4️⃣ Validity checks
        if current_epb <= 0 or np.isinf(current_epb):
            self._clear_ttt(vid)
            return candidate_epb > 0 and np.isfinite(candidate_epb)

        # 5️⃣ Energy hysteresis (unless an emergency path is active)
        if not forced_handoff:
            energy_saving = (current_epb - candidate_epb) / current_epb
            if energy_saving < self.hysteresis:
                return False

        # 6️⃣ 🔽 SINR QUALITY GUARD (don't switch to worse link)
        if candidate_sinr_db is not None and current_sinr_db is not None:
            if candidate_sinr_db < current_sinr_db - 2.0:
                return False

        # 7️⃣ TTT logic (keep existing)
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
