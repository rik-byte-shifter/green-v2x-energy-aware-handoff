from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models.basestation import BaseStation
from src.models.energy import EnergyModel
from src.models.vehicle import Vehicle


class LBAwareRSRPHandoff:
    """
    Load-balancing aware RSRP handoff baseline.

    Based on:
    Hatipoglu et al. (2025), "Handover management in beyond 5G HetNet
    topologies with unbalanced user distribution," Digital Communications and
    Networks, 11(2), 465-472. DOI: 10.1016/j.dcan.2024.05.005

    Algorithm summary:
    - A3-style RSRP trigger with HOM and TTT
    - Candidate ordering by RSRP descending
    - Availability states from BS load:
        available: load < alpha
        semi_available: alpha <= load < 1.0
        unavailable: load >= 1.0
    - Try best three candidates first, then all remaining candidates
    """

    def __init__(
        self,
        *,
        hom_db: float = 3.0,
        ttt_s: float = 0.1,  # Table 2: 100 ms
        load_threshold_alpha: float = 0.8,  # beta operating point (guideline default)
        min_time_since_last_handoff_s: float = 4.0,
    ):
        self.hom_db = hom_db
        self.ttt_s = ttt_s
        self.alpha = load_threshold_alpha
        self.min_time_since_last_handoff_s = min_time_since_last_handoff_s

        self.energy_model = EnergyModel()
        self.total_handoffs = 0
        self.handoff_history = []
        self._ttt_candidate: Dict[int, int] = {}
        self._ttt_since: Dict[int, float] = {}

    def reset_statistics(self):
        self.total_handoffs = 0
        self.handoff_history = []
        self._ttt_candidate.clear()
        self._ttt_since.clear()

    def _bs_availability(self, bs: BaseStation) -> str:
        load = bs.get_load()
        if load < self.alpha:
            return "available"
        if load < 1.0:
            return "semi_available"
        return "unavailable"

    def select_best_bs(
        self,
        vehicle: Vehicle,
        base_stations: List[BaseStation],
        *,
        link_metrics_getter: Optional[
            Callable[[Vehicle, BaseStation, bool], Optional[Dict[str, Any]]]
        ] = None,
    ) -> Tuple[Optional[BaseStation], dict]:
        candidates = []
        for bs in base_stations:
            if not bs.is_in_coverage(vehicle.x, vehicle.y):
                continue

            if link_metrics_getter is not None:
                row = link_metrics_getter(vehicle, bs, False)
                if row is None:
                    continue
                rx_power = float(row.get("rx_power", float("-inf")))
                snr = float(row.get("snr", float("-inf")))
                epb = float(row.get("energy_per_bit", float("inf")))
                data_rate = float(row.get("data_rate", 0.0))
            else:
                distance = bs.distance_to(vehicle.x, vehicle.y)
                tx_power = bs.calculate_tx_power_required_for_target_rx(distance)
                rx_power = bs.calculate_received_power(vehicle.x, vehicle.y, tx_power)
                snr = float("-inf")
                epb = float("inf")
                data_rate = 0.0

            candidates.append(
                {
                    "bs": bs,
                    "rx_power": rx_power,
                    "snr": snr,
                    "energy_per_bit": epb,
                    "data_rate": data_rate,
                    "load": float(bs.get_load()),
                    "availability": self._bs_availability(bs),
                }
            )

        if not candidates:
            return None, {}

        candidates.sort(key=lambda x: x["rx_power"], reverse=True)
        top3 = candidates[:3]

        # Stage 1: available within top-3 by RSRP
        for c in top3:
            if c["availability"] == "available" and c["bs"].has_capacity():
                return c["bs"], c

        # Stage 2: semi-available within top-3 by RSRP
        for c in top3:
            if c["availability"] == "semi_available" and c["bs"].has_capacity():
                return c["bs"], c

        # Stage 3: search all remaining by RSRP for available/semi-available
        for c in candidates[3:]:
            if c["availability"] in ("available", "semi_available") and c["bs"].has_capacity():
                return c["bs"], c

        return None, {}

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
            distance = bs.distance_to(vehicle.x, vehicle.y)
            tx_power = bs.calculate_tx_power_required_for_target_rx(distance)
            return float(bs.calculate_received_power(vehicle.x, vehicle.y, tx_power))
        row = link_metrics_getter(vehicle, bs, False)
        if row is None:
            return None
        return float(row.get("rx_power", float("-inf")))

    def should_handoff(
        self,
        vehicle: Vehicle,
        current_bs: BaseStation,
        candidate_bs: BaseStation,
        current_rx_power: float,
        candidate_rx_power: float,
        *,
        current_time: Optional[float] = None,
    ) -> bool:
        vid = vehicle.vehicle_id

        # Immediate handoff if serving BS is fully loaded.
        if self._bs_availability(current_bs) == "unavailable":
            self._ttt_candidate.pop(vid, None)
            self._ttt_since.pop(vid, None)
            return True

        if current_bs.bs_id == candidate_bs.bs_id:
            self._ttt_candidate.pop(vid, None)
            self._ttt_since.pop(vid, None)
            return False

        if current_time is not None:
            dt = current_time - vehicle.state.last_handoff_time
            if dt < self.min_time_since_last_handoff_s:
                return False

        # A3 trigger condition in dBm space.
        if candidate_rx_power <= current_rx_power + self.hom_db:
            self._ttt_candidate.pop(vid, None)
            self._ttt_since.pop(vid, None)
            return False

        # TTT logic
        if current_time is None or self.ttt_s <= 0.0:
            return True

        cid = candidate_bs.bs_id
        if self._ttt_candidate.get(vid) != cid:
            self._ttt_candidate[vid] = cid
            self._ttt_since[vid] = current_time
            return False

        if current_time - self._ttt_since[vid] >= self.ttt_s:
            self._ttt_candidate.pop(vid, None)
            self._ttt_since.pop(vid, None)
            return True

        return False

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
            "paper_reference": "10.1016/j.dcan.2024.05.005",
            "parameters": {
                "hom_db": self.hom_db,
                "ttt_s": self.ttt_s,
                "alpha": self.alpha,
            },
        }
