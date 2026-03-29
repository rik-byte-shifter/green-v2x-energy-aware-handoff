import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.models.energy import EnergyModel


@dataclass
class VehicleState:
    """Track vehicle communication state"""
    connected_bs_id: Optional[int] = None
    tx_power: float = 0.0
    rx_power: float = 0.0
    data_rate: float = 0.0
    energy_consumed: float = 0.0
    bits_transmitted: float = 0.0
    handoff_count: int = 0
    last_handoff_time: float = -1e9


class Vehicle:
    """
    Vehicular Node for V2X Communication

    Parameters:
    - vehicle_id: Unique identifier
    - x, y: Initial position (meters)
    - speed: Velocity (m/s)
    - direction: Movement direction (radians)
    """

    def __init__(self, vehicle_id: int, x: float, y: float,
                 speed: float = 20.0, direction: float = 0.0):
        self.vehicle_id = vehicle_id
        self.x = x
        self.y = y
        self.speed = speed  # m/s
        self.direction = direction  # radians
        self.state = VehicleState()

        # Communication parameters
        self.max_tx_power = 0.2  # 200 mW
        self.min_tx_power = 0.001  # 1 mW
        self.antenna_gain = 0  # dBi

    def move(self, delta_time: float = 1.0, boundary: tuple = (0, 2000)):
        """
        Update vehicle position based on speed and direction

        Parameters:
        - delta_time: Time step (seconds)
        - boundary: Simulation area boundaries (min, max)
        """
        self.x += self.speed * np.cos(self.direction) * delta_time
        self.y += self.speed * np.sin(self.direction) * delta_time

        min_bound, max_bound = boundary
        if self.x < min_bound or self.x > max_bound:
            self.direction = np.pi - self.direction
            self.x = np.clip(self.x, min_bound, max_bound)
        if self.y < min_bound or self.y > max_bound:
            self.direction = -self.direction
            self.y = np.clip(self.y, min_bound, max_bound)

    def get_position(self) -> tuple:
        """Return current (x, y) position"""
        return (self.x, self.y)

    def distance_to(self, x: float, y: float) -> float:
        """Calculate distance to a point"""
        return np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def update_energy(self, tx_power: float, duration: float, data_rate: float,
                      energy_model: Optional["EnergyModel"] = None):
        """
        Track energy consumption. If energy_model is given, uses total TX-chain power
        (PA + circuits); otherwise RF output only (tx_power * duration).
        """
        if energy_model is not None:
            energy = energy_model.calculate_total_power(tx_power, 'transmit') * duration
        else:
            energy = tx_power * duration
        bits = data_rate * duration

        self.state.energy_consumed += energy
        self.state.bits_transmitted += bits
        self.state.tx_power = tx_power
        self.state.data_rate = data_rate

    def calculate_energy_per_bit(self) -> float:
        """Calculate energy efficiency (Joules per bit)"""
        if self.state.bits_transmitted == 0:
            return float('inf')
        return self.state.energy_consumed / self.state.bits_transmitted

    def reset_stats(self):
        """Reset statistics for new simulation run"""
        self.state = VehicleState()

    def __repr__(self):
        return f"Vehicle(id={self.vehicle_id}, pos=({self.x:.1f}, {self.y:.1f}), " \
               f"speed={self.speed:.1f} m/s)"
