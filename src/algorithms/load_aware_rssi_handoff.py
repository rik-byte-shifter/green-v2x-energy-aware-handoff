from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models.basestation import BaseStation
from src.models.energy import EnergyModel
from src.models.vehicle import Vehicle


class LoadAwareRSSIHandoff:
    """
    Semi-smart baseline: maximize (RSSI - alpha * load).
    """

    def __init__(self, hysteresis_db: float = 1.5, load_penalty_alpha_db: float = 6.0):
        self.hysteresis_db = hysteresis_db
        self.load_penalty_alpha_db = load_penalty_alpha_db
        self.energy_model = EnergyModel()

        self.total_handoffs = 0
        self.handoff_history = []

    def reset_statistics(self):
        self.total_handoffs = 0
        self.handoff_history = []

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
        best_score = float("-inf")
        best_rssi = float("-inf")
        best_load = 1.0
        best_distance = float("inf")

        for bs in base_stations:
            if link_metrics_getter is not None:
                row = link_metrics_getter(vehicle, bs, True)
                if row is None:
                    continue
                distance = row["distance"]
                rssi = row["rx_power"]
                load = row["load"]
            else:
                if not bs.is_in_coverage(vehicle.x, vehicle.y):
                    continue
                if not bs.has_capacity():
                    continue
                distance = bs.distance_to(vehicle.x, vehicle.y)
                tx_link = bs.calculate_tx_power_required_for_target_rx(distance)
                rssi = bs.calculate_received_power(vehicle.x, vehicle.y, tx_link)
                load = bs.get_load()

            score = rssi - self.load_penalty_alpha_db * load
            if score > best_score:
                best_score = score
                best_rssi = rssi
                best_load = load
                best_bs = bs
                best_distance = distance

        return best_bs, {
            "score": best_score,
            "rssi": best_rssi,
            "load": best_load,
            "distance": best_distance,
        }

    def should_handoff(self, current_score: float, candidate_score: float) -> bool:
        return candidate_score > (current_score + self.hysteresis_db)

    def execute_handoff(
        self, vehicle: Vehicle, old_bs: Optional[BaseStation], new_bs: BaseStation
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
        n_vehicles = len(set(h["vehicle_id"] for h in self.handoff_history))
        return {
            "total_handoffs": self.total_handoffs,
            "avg_handoffs_per_vehicle": self.total_handoffs / max(1, n_vehicles),
        }
