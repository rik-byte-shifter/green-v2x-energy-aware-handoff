import numpy as np
from dataclasses import dataclass
from statistics import NormalDist
from typing import List

from src.models.channel import apply_log_normal_shadowing


@dataclass
class BSConfig:
    """Base Station Configuration"""
    coverage_radius: float = 300.0  # meters

    # FIX: max_capacity raised from 20 -> 100.
    #
    # WHY THIS MATTERS:
    # The old default of 20 caused a hard block in the scaling experiment.
    # At 200 vehicles / 16 BSs, the average load is ~12.5 vehicles per BS.
    # With max_capacity=20, has_capacity() returns False once a BS has 20
    # vehicles, so many vehicles during initial association find all nearby BSs
    # "full" and cannot connect. Every algorithm then falls back identically,
    # producing 0% energy saving with zero variance -- not a real result.
    #
    # Real cellular base stations serve hundreds of UEs simultaneously.
    # Setting max_capacity=100 is realistic and keeps has_capacity() True
    # at all vehicle counts tested (20-200), so algorithm differentiation
    # is preserved throughout the scalability sweep.
    #
    # Effect on load metric:
    #   get_load() = connected / max_capacity
    #   At 200 vehicles / 16 BSs: ~12.5/100 = 0.125 load
    #   load_penalty in load_aware_rssi = 6.0 * 0.125 = 0.75 dB  (meaningful)
    #   Example paper baseline (20 veh / 16 BSs, balanced): ~1.25/100 load
    max_capacity: int = 100

    frequency: float = 2.4e9  # 2.4 GHz
    bandwidth: float = 20e6   # 20 MHz
    noise_figure: float = 5.0 # dB
    # Log-normal shadowing on received power (dB); 0 disables
    shadowing_std_db: float = 8.0
    # Distance loss scaling exponent (n in 10*n*log10(d)).
    path_loss_exponent: float = 2.0
    # Extra attenuation applied per km (dB/km). 0 disables.
    rain_attenuation_db_per_km: float = 0.0
    # TX link-budget settings (percentile reliability design under shadowing).
    target_rx_power_dbm: float = -90.0
    shadowing_reliability: float = 0.95
    tx_power_min_watts: float = 0.005
    tx_power_max_watts: float = 0.5


class BaseStation:
    """
    5G/6G Base Station for V2X Communication

    Parameters:
    - bs_id: Unique identifier
    - x, y: Position (meters)
    - config: BS configuration
    """

    def __init__(self, bs_id: int, x: float, y: float,
                 config: BSConfig = None):
        self.bs_id = bs_id
        self.x = x
        self.y = y
        self.config = config or BSConfig()

        self.connected_vehicles: List[int] = []
        self.total_served_vehicles = 0
        self.total_data_transmitted = 0.0

    def distance_to(self, x: float, y: float) -> float:
        """Calculate Euclidean distance to a point"""
        return np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def is_in_coverage(self, x: float, y: float) -> bool:
        """Check if point is within coverage area"""
        return self.distance_to(x, y) <= self.config.coverage_radius

    def calculate_path_loss(self, distance: float) -> float:
        """
        Calculate path loss using simplified model.

        Path Loss (dB) = 10*n*log10(d) + 20*log10(f) - 147.55 + rain_att(d)
        where:
          - d is in meters
          - rain_att(d) = rain_attenuation_db_per_km * (d/1000)
        """
        if distance < 1:
            distance = 1

        frequency = self.config.frequency
        path_loss = (
            10 * self.config.path_loss_exponent * np.log10(distance)
            + 20 * np.log10(frequency)
            - 147.55
        )
        d_km = float(distance) / 1000.0
        path_loss += self.config.rain_attenuation_db_per_km * d_km
        return path_loss

    def calculate_tx_power_required_for_target_rx(
        self,
        distance_m: float,
        target_rx_power_dbm: float | None = None,
        shadowing_reliability: float | None = None,
    ) -> float:
        """
        Compute required TX power (W) with percentile shadowing margin:

            P_tx_required(dBm) = P_rx_target(dBm) + PL_mean(d) + M
            M = z_p * sigma_shadowing_db

        where z_p is the Gaussian quantile for reliability p.
        """
        if target_rx_power_dbm is None:
            target_rx_power_dbm = self.config.target_rx_power_dbm
        if shadowing_reliability is None:
            shadowing_reliability = self.config.shadowing_reliability

        p = float(np.clip(shadowing_reliability, 1e-6, 1.0 - 1e-6))
        sigma_db = max(0.0, float(self.config.shadowing_std_db))
        z_p = NormalDist().inv_cdf(p)
        margin_db = z_p * sigma_db

        path_loss_db = self.calculate_path_loss(distance_m)
        tx_power_dbm = float(target_rx_power_dbm) + path_loss_db + margin_db

        tx_power_watts = 10 ** ((tx_power_dbm - 30.0) / 10.0)
        tx_power_watts = max(tx_power_watts, self.config.tx_power_min_watts)
        tx_power_watts = min(tx_power_watts, self.config.tx_power_max_watts)

        return float(tx_power_watts)

    def calculate_received_power(self, x: float, y: float,
                                  tx_power_watts: float) -> float:
        """
        Calculate received power at vehicle location.

        Returns:
        - Received power in dBm
        """
        distance = self.distance_to(x, y)
        tx_power_dbm = 10 * np.log10(tx_power_watts * 1000)
        path_loss = self.calculate_path_loss(distance)
        rx_power = tx_power_dbm - path_loss
        return apply_log_normal_shadowing(rx_power, self.config.shadowing_std_db)

    def has_capacity(self) -> bool:
        """Check if BS can accept more vehicles"""
        return len(self.connected_vehicles) < self.config.max_capacity

    def add_vehicle(self, vehicle_id: int):
        """Add vehicle to connected list"""
        if vehicle_id not in self.connected_vehicles:
            self.connected_vehicles.append(vehicle_id)
            self.total_served_vehicles += 1

    def remove_vehicle(self, vehicle_id: int):
        """Remove vehicle from connected list"""
        if vehicle_id in self.connected_vehicles:
            self.connected_vehicles.remove(vehicle_id)

    def get_load(self) -> float:
        """Get current load as fraction of max_capacity"""
        return len(self.connected_vehicles) / self.config.max_capacity

    def __repr__(self):
        return (
            f"BS(id={self.bs_id}, pos=({self.x:.1f}, {self.y:.1f}), "
            f"vehicles={len(self.connected_vehicles)}/{self.config.max_capacity})"
        )
