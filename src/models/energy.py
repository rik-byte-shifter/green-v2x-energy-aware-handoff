import numpy as np
from dataclasses import dataclass


@dataclass
class EnergyParams:
    """Energy consumption parameters"""
    p_tx_circuit: float = 0.1
    p_rx_circuit: float = 0.05
    p_idle: float = 0.01
    p_sleep: float = 0.001

    pa_efficiency: float = 0.35

    p_baseband: float = 0.02
    p_cooling: float = 0.005


class EnergyModel:
    """
    Energy Consumption Model for V2X Communication
    """

    def __init__(self, params: EnergyParams = None):
        self.params = params or EnergyParams()

    def calculate_tx_power_required(self, distance: float,
                                    min_rx_power: float = -90.0,
                                    path_loss_exponent: float = 3.0) -> float:
        path_loss = 10 * path_loss_exponent * np.log10(distance)

        tx_power_dbm = min_rx_power + path_loss

        tx_power_watts = 10 ** ((tx_power_dbm - 30) / 10.0)

        tx_power_watts = max(tx_power_watts, 0.001)
        tx_power_watts = min(tx_power_watts, 0.2)

        return tx_power_watts

    def calculate_total_power(self, tx_power: float,
                             mode: str = 'transmit') -> float:
        if mode == 'transmit':
            pa_power = tx_power / self.params.pa_efficiency
            total = pa_power + self.params.p_tx_circuit + \
                    self.params.p_baseband + self.params.p_cooling

        elif mode == 'receive':
            total = self.params.p_rx_circuit + \
                    self.params.p_baseband

        elif mode == 'idle':
            total = self.params.p_idle

        elif mode == 'sleep':
            total = self.params.p_sleep

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return total

    def calculate_energy_per_bit(self, tx_power: float,
                                 data_rate: float,
                                 packet_size: int = 1000,
                                 mode: str = 'transmit') -> float:
        if data_rate <= 0:
            return float('inf')
        tx_time = packet_size / data_rate

        total_power = self.calculate_total_power(tx_power, mode)

        total_energy = total_power * tx_time

        energy_per_bit = total_energy / packet_size

        return energy_per_bit

    def calculate_communication_energy(self, tx_power: float,
                                       data_rate: float,
                                       duration: float,
                                       mode: str = 'transmit') -> dict:
        total_power = self.calculate_total_power(tx_power, mode)
        total_energy = total_power * duration
        bits_transmitted = data_rate * duration

        energy_per_bit = total_energy / bits_transmitted if bits_transmitted > 0 else 0

        return {
            'total_power': total_power,
            'total_energy': total_energy,
            'bits_transmitted': bits_transmitted,
            'energy_per_bit': energy_per_bit,
            'pa_power': tx_power / self.params.pa_efficiency,
            'circuit_power': total_power - (tx_power / self.params.pa_efficiency)
        }
