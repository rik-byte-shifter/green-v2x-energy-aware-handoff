from typing import List, Optional, Tuple

from src.models.vehicle import Vehicle
from src.models.basestation import BaseStation


class NaiveNearestHandoff:
    """
    Baseline: always associate with the geographically nearest in-coverage BS.
    No RSSI hysteresis and no energy metric — highlights cost of naive attachment.
    """

    def __init__(self):
        self.total_handoffs = 0
        self.handoff_history = []

    def reset_statistics(self):
        self.total_handoffs = 0
        self.handoff_history = []

    def select_best_bs(
        self,
        vehicle: Vehicle,
        base_stations: List[BaseStation],
    ) -> Tuple[Optional[BaseStation], dict]:
        candidates: List[BaseStation] = []
        for bs in base_stations:
            if not bs.is_in_coverage(vehicle.x, vehicle.y):
                continue
            if not bs.has_capacity():
                continue
            candidates.append(bs)

        if not candidates:
            return None, {}

        best = min(
            candidates,
            key=lambda bs: (bs.distance_to(vehicle.x, vehicle.y), bs.bs_id),
        )
        d = best.distance_to(vehicle.x, vehicle.y)
        return best, {"distance": d, "bs_id": best.bs_id}

    def should_handoff(
        self,
        vehicle: Vehicle,
        current_bs: BaseStation,
        candidate_bs: BaseStation,
    ) -> bool:
        return candidate_bs.bs_id != current_bs.bs_id

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
        n_vehicles = len(set(h["vehicle_id"] for h in self.handoff_history))
        return {
            "total_handoffs": self.total_handoffs,
            "avg_handoffs_per_vehicle": self.total_handoffs / max(1, n_vehicles),
        }
