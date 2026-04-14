from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models.basestation import BaseStation
from src.models.energy import EnergyModel
from src.models.vehicle import Vehicle


class EnhancedUplinkLiteratureHandoff:
    """
    UL-RSRP + load-gated literature baseline:
    Jon et al., Wireless Networks 2024 (10.1007/s11276-023-03547-7).

    Selection:
      - Candidate cells must satisfy load <= alpha (load_threshold) and capacity.
      - Among candidates, select highest UL-RSRP (proxy: cached rx_power at BS).

    Trigger (A3-style):
      - candidate_rsrp > serving_rsrp + HOM, held for TTT.
    """

    def __init__(
        self,
        *,
        hom_db: float = 1.0,
        ttt_s: float = 0.032,
        load_threshold: float = 0.8,
        min_time_since_last_handoff_s: float = 4.0,
    ):
        self.hom_db = hom_db
        self.ttt_s = ttt_s
        self.load_threshold = load_threshold
        self.min_time_since_last_handoff_s = min_time_since_last_handoff_s

        self.energy_model = EnergyModel()
        self.total_handoffs = 0
        self.handoff_history = []
        self._ttt_candidate_bs_id: Dict[int, int] = {}
        self._ttt_since: Dict[int, float] = {}

    def reset_statistics(self):
        self.total_handoffs = 0
        self.handoff_history = []
        self._ttt_candidate_bs_id.clear()
        self._ttt_since.clear()

    def _clear_ttt(self, vehicle_id: int):
        self._ttt_candidate_bs_id.pop(vehicle_id, None)
        self._ttt_since.pop(vehicle_id, None)

    def _rsrp_row(
        self,
        vehicle: Vehicle,
        bs: BaseStation,
        *,
        require_capacity: bool = True,
        link_metrics_getter: Optional[
            Callable[[Vehicle, BaseStation, bool], Optional[Dict[str, Any]]]
        ] = None,
    ) -> Optional[Dict[str, float]]:
        if link_metrics_getter is None:
            if not bs.is_in_coverage(vehicle.x, vehicle.y):
                return None
            if bs.get_load() > self.load_threshold:
                return None
            if require_capacity and not bs.has_capacity():
                return None
            distance = bs.distance_to(vehicle.x, vehicle.y)
            tx = bs.calculate_tx_power_required_for_target_rx(distance)
            rx = bs.calculate_received_power(vehicle.x, vehicle.y, tx)
        else:
            row = link_metrics_getter(vehicle, bs, require_capacity)
            if row is None:
                return None
            if float(row.get("load", bs.get_load())) > self.load_threshold:
                return None
            distance = float(row["distance"])
            tx = float(row["tx_power"])
            rx = float(row["rx_power"])

        return {
            "rx_power": float(rx),
            "snr": float("-inf") if link_metrics_getter is None else float(row.get("snr", float("-inf"))),
            "data_rate": 0.0 if link_metrics_getter is None else float(row.get("data_rate", 0.0)),
            "energy_per_bit": float("inf") if link_metrics_getter is None else float(row.get("energy_per_bit", float("inf"))),
            "load": float(bs.get_load()),
            "distance": float(distance),
            "tx_power": float(tx),
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
        best_bs: Optional[BaseStation] = None
        best_info: Optional[Dict[str, float]] = None
        best_rx_power = float("-inf")

        for bs in base_stations:
            info = self._rsrp_row(
                vehicle,
                bs,
                require_capacity=True,
                link_metrics_getter=link_metrics_getter,
            )
            if info is None:
                continue
            if info["rx_power"] > best_rx_power:
                best_rx_power = info["rx_power"]
                best_info = info
                best_bs = bs

        if best_bs is None or best_info is None:
            return None, {}
        return best_bs, best_info

    def link_metric_value(
        self,
        vehicle: Vehicle,
        bs: BaseStation,
        *,
        link_metrics_getter: Optional[
            Callable[[Vehicle, BaseStation, bool], Optional[Dict[str, Any]]]
        ] = None,
    ) -> Optional[float]:
        row = self._rsrp_row(
            vehicle,
            bs,
            require_capacity=False,
            link_metrics_getter=link_metrics_getter,
        )
        if row is None:
            return None
        return float(row["rx_power"])

    def should_handoff(
        self,
        vehicle: Vehicle,
        current_bs: BaseStation,
        candidate_bs: BaseStation,
        current_metric: float,
        candidate_metric: float,
        *,
        current_time: Optional[float] = None,
    ) -> bool:
        # If serving BS exceeds load threshold, force a handoff if possible.
        if current_bs.get_load() > self.load_threshold:
            self._clear_ttt(vehicle.vehicle_id)
            return True

        if current_bs.bs_id == candidate_bs.bs_id:
            self._clear_ttt(vehicle.vehicle_id)
            return False

        if current_time is not None:
            time_since_last = current_time - vehicle.state.last_handoff_time
            if time_since_last < self.min_time_since_last_handoff_s:
                return False

        # A3 condition in RSRP space (both values are dBm).
        if candidate_metric <= current_metric + self.hom_db:
            self._clear_ttt(vehicle.vehicle_id)
            return False

        if current_time is None or self.ttt_s <= 0.0:
            self._clear_ttt(vehicle.vehicle_id)
            return True

        vid = vehicle.vehicle_id
        cid = candidate_bs.bs_id
        if self._ttt_candidate_bs_id.get(vid) != cid:
            self._ttt_candidate_bs_id[vid] = cid
            self._ttt_since[vid] = current_time
            return False

        if current_time - self._ttt_since[vid] >= self.ttt_s:
            self._clear_ttt(vid)
            return True
        return False

    def execute_handoff(
        self,
        vehicle: Vehicle,
        old_bs: Optional[BaseStation],
        new_bs: BaseStation,
    ):
        self._clear_ttt(vehicle.vehicle_id)
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
        n_vehicles = len(set(h["vehicle_id"] for h in self.handoff_history))
        return {
            "total_handoffs": self.total_handoffs,
            "avg_handoffs_per_vehicle": self.total_handoffs / max(1, n_vehicles),
            "paper_reference": "10.1007/s11276-023-03547-7",
            "parameters": {
                "hom_db": self.hom_db,
                "ttt_s": self.ttt_s,
                "load_threshold": self.load_threshold,
            },
        }
