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
                                    path_loss_exponent: float = 3.5) -> float:
        """Minimum TX power (W) to close the link; distance in meters."""
        path_loss = 10 * path_loss_exponent * np.log10(max(distance, 1.0))

        tx_power_dbm = min_rx_power + path_loss

        tx_power_watts = 10 ** ((tx_power_dbm - 30) / 10.0)

        tx_power_watts = max(tx_power_watts, 0.005)
        tx_power_watts = min(tx_power_watts, 0.5)

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


class EnvironmentalMetrics:
    """
    Map communication / device energy (as proxy for grid-drawn energy) to CO2
    using an electricity carbon intensity factor.
    """

    def __init__(self, carbon_intensity_kg_per_kwh: float = 0.5):
        """
        Args:
            carbon_intensity_kg_per_kwh: kg CO2 emitted per kWh of electricity.
                Reference values: global average ~0.5; coal-heavy ~0.8–1.0;
                renewable-heavy ~0.05–0.1.
        """
        self.carbon_intensity = carbon_intensity_kg_per_kwh

    def energy_to_co2(self, energy_joules: float) -> float:
        """Convert electrical energy (J) to CO2 mass (kg)."""
        energy_kwh = energy_joules / 3.6e6
        return energy_kwh * self.carbon_intensity

    def calculate_vehicle_co2_footprint(self, vehicle) -> dict:
        total_energy = vehicle.state.energy_consumed
        co2_kg = self.energy_to_co2(total_energy)
        return {
            'energy_joules': total_energy,
            'co2_kg': co2_kg,
            'co2_grams': co2_kg * 1000,
        }

    def avg_co2_kg_per_vehicle(self, total_co2_kg: float, num_vehicles: int) -> float:
        """Fleet CO2 over the simulation interval, averaged per vehicle (kg)."""
        if num_vehicles <= 0:
            return 0.0
        return total_co2_kg / num_vehicles

    def co2_kg_per_vehicle_per_year(
        self,
        total_co2_kg: float,
        num_vehicles: int,
        simulation_duration_s: float,
        seconds_per_year: float,
    ) -> float:
        """
        Extrapolate average per-vehicle CO2 to an annual figure (kg CO2 / vehicle / year).

        Uses linear scaling: (total_co2/n) * (seconds_per_year / simulation_duration_s).
        This assumes the simulated interval is representative of the same duty cycle
        over the full year.
        """
        if num_vehicles <= 0 or simulation_duration_s <= 0 or seconds_per_year <= 0:
            return 0.0
        return self.avg_co2_kg_per_vehicle(total_co2_kg, num_vehicles) * (
            seconds_per_year / simulation_duration_s
        )
