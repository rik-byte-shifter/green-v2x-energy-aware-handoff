import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models.basestation import BaseStation
from src.models.energy import EnergyModel
from src.models.vehicle import Vehicle


class MDPIEnergyEfficientHandoff:
    """
    Composite-score literature baseline from:
    Abdullah et al. (2024), Journal of Sensor and Actuator Networks, 13(5), 51.

    Paper formula (Section 4.2):
      S = W_RSSI*RSSI + W_SINR*SINR + W_BUE*BUE + W_Econs*Econs + W_MUE*MUE + W_QoS*QoS

    Implementation notes for V2X simulator compatibility:
    - BUE (battery level) is approximated by a TX-power stress proxy.
    - Econs is represented by inverse-normalized energy-per-bit quality.
    - MUE uses movement-direction alignment toward candidate BS.
    - QoS uses normalized data-rate / SINR quality.
    """

    def __init__(
        self,
        *,
        threshold_rssi_dbm: float = -85.0,
        threshold_sinr_db: float = 12.0,
        handover_threshold: float = 0.75,
        critical_battery_proxy: float = 0.3,
        tx_power_max_watts: float = 0.5,
        score_margin: float = 0.02,
        min_time_since_last_handoff_s: float = 4.0,
        w_rssi: float = 1.0 / 6.0,
        w_sinr: float = 1.0 / 6.0,
        w_bue: float = 1.0 / 6.0,
        w_econs: float = 1.0 / 6.0,
        w_mue: float = 1.0 / 6.0,
        w_qos: float = 1.0 / 6.0,
    ):
        self.threshold_rssi_dbm = threshold_rssi_dbm
        self.threshold_sinr_db = threshold_sinr_db
        self.handover_threshold = handover_threshold
        self.critical_battery_proxy = critical_battery_proxy
        self.tx_power_max_watts = tx_power_max_watts
        self.score_margin = score_margin
        self.min_time_since_last_handoff_s = min_time_since_last_handoff_s

        wsum = w_rssi + w_sinr + w_bue + w_econs + w_mue + w_qos
        if wsum <= 0:
            raise ValueError("Composite weights must sum to a positive value")
        self.w_rssi = w_rssi / wsum
        self.w_sinr = w_sinr / wsum
        self.w_bue = w_bue / wsum
        self.w_econs = w_econs / wsum
        self.w_mue = w_mue / wsum
        self.w_qos = w_qos / wsum

        self.energy_model = EnergyModel()
        self.total_handoffs = 0
        self.handoff_history = []

    def reset_statistics(self):
        self.total_handoffs = 0
        self.handoff_history = []

    @staticmethod
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def _norm_rssi(self, rssi_dbm: float) -> float:
        # practical range for cellular-style link quality
        return self._clamp01((rssi_dbm + 120.0) / 60.0)

    def _norm_sinr(self, sinr_db: float) -> float:
        # maps roughly [-10, 30] dB to [0,1]
        return self._clamp01((sinr_db + 10.0) / 40.0)

    def _bue_proxy(self, tx_power_w: float) -> float:
        # lower TX stress implies healthier battery equivalent
        frac = tx_power_w / max(1e-9, self.tx_power_max_watts)
        return self._clamp01(1.0 - frac)

    @staticmethod
    def _direction_alignment(vehicle: Vehicle, bs: BaseStation) -> float:
        rel_x = bs.x - vehicle.x
        rel_y = bs.y - vehicle.y
        radial_angle = math.atan2(rel_y, rel_x)
        d = (vehicle.direction - radial_angle + math.pi) % (2.0 * math.pi) - math.pi
        return max(0.0, 1.0 - abs(d) / math.pi)

    def _qos_score(self, sinr_db: float, data_rate_bps: float) -> float:
        sinr_part = self._norm_sinr(sinr_db)
        rate_part = self._clamp01(data_rate_bps / 1e6)  # paper's minimum bandwidth target: 1 Mbps
        return 0.5 * sinr_part + 0.5 * rate_part

    def _econs_score(self, energy_per_bit: float) -> float:
        # lower EPB is better; map to quality score
        if not math.isfinite(energy_per_bit) or energy_per_bit <= 0:
            return 0.0
        # 1e-9 J/bit => high quality, >=1e-6 => poor quality
        logv = math.log10(energy_per_bit)
        return self._clamp01(( -6.0 - logv) / 3.0)

    def _candidate_row(
        self,
        vehicle: Vehicle,
        bs: BaseStation,
        *,
        link_metrics_getter: Optional[
            Callable[[Vehicle, BaseStation, bool], Optional[Dict[str, Any]]]
        ] = None,
    ) -> Optional[Dict[str, float]]:
        if link_metrics_getter is None:
            if not bs.is_in_coverage(vehicle.x, vehicle.y) or not bs.has_capacity():
                return None
            d = bs.distance_to(vehicle.x, vehicle.y)
            tx = bs.calculate_tx_power_required_for_target_rx(d)
            rssi = bs.calculate_received_power(vehicle.x, vehicle.y, tx)
            sinr = rssi
            data_rate = max(0.0, 20e6 * math.log2(1.0 + 10.0 ** (sinr / 10.0)))
            epb = float("inf")
        else:
            row = link_metrics_getter(vehicle, bs, True)
            if row is None:
                return None
            tx = float(row["tx_power"])
            rssi = float(row["rx_power"])
            sinr = float(row.get("snr", rssi))
            data_rate = float(row.get("data_rate", 0.0))
            epb = float(row.get("energy_per_bit", float("inf")))

        bue = self._bue_proxy(tx)
        mue = self._direction_alignment(vehicle, bs)
        qos = self._qos_score(sinr, data_rate)
        econs = self._econs_score(epb)
        score = (
            self.w_rssi * self._norm_rssi(rssi)
            + self.w_sinr * self._norm_sinr(sinr)
            + self.w_bue * bue
            + self.w_econs * econs
            + self.w_mue * mue
            + self.w_qos * qos
        )
        return {
            "score": float(score),
            "rx_power": float(rssi),
            "snr": float(sinr),
            "data_rate": float(data_rate),
            "energy_per_bit": float(epb),
            "bue_proxy": float(bue),
            "econs_score": float(econs),
            "mue_score": float(mue),
            "qos_score": float(qos),
        }

    def select_best_bs(
        self,
        vehicle: Vehicle,
        base_stations: List[BaseStation],
        *,
        link_metrics_getter: Optional[
            Callable[[Vehicle, BaseStation, bool], Optional[Dict[str, Any]]]
        ] = None,
    ) -> Tuple[Optional[BaseStation], dict]:
        best_bs = None
        best = None
        best_score = float("-inf")
        for bs in base_stations:
            row = self._candidate_row(vehicle, bs, link_metrics_getter=link_metrics_getter)
            if row is None:
                continue
            if row["score"] > best_score:
                best_score = row["score"]
                best = row
                best_bs = bs
        if best_bs is None or best is None:
            return None, {}
        return best_bs, best

    def link_metric_value(
        self,
        vehicle: Vehicle,
        bs: BaseStation,
        *,
        link_metrics_getter: Optional[
            Callable[[Vehicle, BaseStation, bool], Optional[Dict[str, Any]]]
        ] = None,
    ) -> Optional[float]:
        if not bs.is_in_coverage(vehicle.x, vehicle.y):
            return None
        if link_metrics_getter is None:
            d = bs.distance_to(vehicle.x, vehicle.y)
            tx = bs.calculate_tx_power_required_for_target_rx(d)
            rssi = bs.calculate_received_power(vehicle.x, vehicle.y, tx)
            sinr = rssi
            data_rate = max(0.0, 20e6 * math.log2(1.0 + 10.0 ** (sinr / 10.0)))
            epb = float("inf")
        else:
            row = link_metrics_getter(vehicle, bs, False)
            if row is None:
                return None
            tx = float(row["tx_power"])
            rssi = float(row["rx_power"])
            sinr = float(row.get("snr", rssi))
            data_rate = float(row.get("data_rate", 0.0))
            epb = float(row.get("energy_per_bit", float("inf")))
        bue = self._bue_proxy(tx)
        mue = self._direction_alignment(vehicle, bs)
        qos = self._qos_score(sinr, data_rate)
        econs = self._econs_score(epb)
        score = (
            self.w_rssi * self._norm_rssi(rssi)
            + self.w_sinr * self._norm_sinr(sinr)
            + self.w_bue * bue
            + self.w_econs * econs
            + self.w_mue * mue
            + self.w_qos * qos
        )
        return float(score)

    def should_handoff(
        self,
        vehicle: Vehicle,
        current_bs: BaseStation,
        candidate_bs: BaseStation,
        current_score: float,
        candidate_score: float,
        *,
        current_time: Optional[float] = None,
        current_rssi_dbm: Optional[float] = None,
        current_sinr_db: Optional[float] = None,
        current_bue_proxy: Optional[float] = None,
    ) -> bool:
        if current_bs.bs_id == candidate_bs.bs_id:
            return False
        if current_time is not None:
            dt = current_time - vehicle.state.last_handoff_time
            if dt < self.min_time_since_last_handoff_s:
                return False

        # Trigger condition adapted from Algorithm 1 gate.
        trigger = False
        if current_rssi_dbm is not None and current_rssi_dbm < self.threshold_rssi_dbm:
            trigger = True
        if current_sinr_db is not None and current_sinr_db < self.threshold_sinr_db:
            trigger = True
        if current_bue_proxy is not None and current_bue_proxy < self.critical_battery_proxy:
            trigger = True
        if not trigger:
            return False

        if candidate_score < self.handover_threshold:
            return False
        return candidate_score > (current_score + self.score_margin)

    def execute_handoff(
        self,
        vehicle: Vehicle,
        old_bs: Optional[BaseStation],
        new_bs: BaseStation,
    ):
        if old_bs is not None:
            old_bs.remove_vehicle(vehicle.vehicle_id)
        new_bs.add_vehicle(vehicle.vehicle_id)
        vehicle.state.connected_bs_id = new_bs.bs_id
        vehicle.state.handoff_count += 1
        self.total_handoffs += 1
        self.handoff_history.append(
            {
                "vehicle_id": vehicle.vehicle_id,
                "from_bs": old_bs.bs_id if old_bs else None,
                "to_bs": new_bs.bs_id,
                "timestamp": len(self.handoff_history),
            }
        )

    def get_statistics(self) -> dict:
        n = len(set(h["vehicle_id"] for h in self.handoff_history))
        return {
            "total_handoffs": self.total_handoffs,
            "avg_handoffs_per_vehicle": self.total_handoffs / max(1, n),
            "paper_reference": "J Sens Actuator Netw 2024, 13(5), 51",
            "parameters": {
                "threshold_rssi_dbm": self.threshold_rssi_dbm,
                "threshold_sinr_db": self.threshold_sinr_db,
                "handover_threshold": self.handover_threshold,
            },
        }
