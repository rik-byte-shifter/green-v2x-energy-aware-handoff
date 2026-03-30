import numpy as np
from dataclasses import dataclass
from typing import List

from src.models.channel import apply_log_normal_shadowing


@dataclass
class BSConfig:
    """Base Station Configuration"""
    coverage_radius: float = 300.0  # meters
    max_capacity: int = 20  # max vehicles
    frequency: float = 2.4e9  # 2.4 GHz
    bandwidth: float = 20e6  # 20 MHz
    noise_figure: float = 5.0  # dB
    # Log-normal shadowing on received power (dB); 0 disables
    shadowing_std_db: float = 8.0
    # Distance loss scaling exponent (n in 10*n*log10(d)).
    path_loss_exponent: float = 2.0
    # Extra attenuation applied per km (dB/km). 0 disables.
    rain_attenuation_db_per_km: float = 0.0


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
        Calculate path loss using simplified model

        Path Loss (dB) = 10*n*log10(d) + 20*log10(f) - 147.55 + rain_att(d)
        where:
          - d is in meters (distance term is unit-consistent with the original code),
          - rain_att(d) = rain_attenuation_db_per_km * (d/1000).

        This keeps the simulator stable (defaults match the legacy exponent-2 term)
        while allowing weather to change the distance exponent and add attenuation.
        """
        if distance < 1:
            distance = 1

        frequency = self.config.frequency
        path_loss = (
            10 * self.config.path_loss_exponent * np.log10(distance)
            + 20 * np.log10(frequency)
            - 147.55
        )

        # Extra weather loss term.
        d_km = float(distance) / 1000.0
        path_loss += self.config.rain_attenuation_db_per_km * d_km
        return path_loss

    def calculate_received_power(self, x: float, y: float,
                                  tx_power_watts: float) -> float:
        """
        Calculate received power at vehicle location

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
        """Get current load as percentage"""
        return len(self.connected_vehicles) / self.config.max_capacity

    def __repr__(self):
        return f"BS(id={self.bs_id}, pos=({self.x:.1f}, {self.y:.1f}), " \
               f"vehicles={len(self.connected_vehicles)})"
