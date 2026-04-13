from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional


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
    Energy consumption model for the vehicular uplink chain (PA, circuits, baseband).

    When ``use_calibration`` is True, order-one scale factors from
    :class:`src.utils.hardware_validation.EnergyModelValidator` nudge total power
    toward published BS/OBU anchors (see ``calibration_factors``).
    """

    def __init__(
        self,
        params: Optional[EnergyParams] = None,
        *,
        use_calibration: bool = True,
    ):
        self.params = params or EnergyParams()
        self.use_calibration = use_calibration
        self.calibration_factors = {"bs": 1.0, "obu": 1.0}
        if use_calibration:
            from src.utils.hardware_validation import EnergyModelValidator

            v = EnergyModelValidator()
            self.calibration_factors["bs"] = v.get_calibration_factor("5G_BS_macro")
            self.calibration_factors["obu"] = v.get_calibration_factor("OBU_802.11p")

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

    def calculate_total_power(
        self,
        tx_power: float,
        mode: str = "transmit",
        device_type: str = "obu",
    ) -> float:
        """Instantaneous chain power (W); optional calibration by ``device_type`` (``bs`` / ``obu``)."""
        if mode == "transmit":
            pa_power = tx_power / self.params.pa_efficiency
            total = (
                pa_power
                + self.params.p_tx_circuit
                + self.params.p_baseband
                + self.params.p_cooling
            )

        elif mode == "receive":
            total = self.params.p_rx_circuit + self.params.p_baseband

        elif mode == "idle":
            total = self.params.p_idle

        elif mode == "sleep":
            total = self.params.p_sleep

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if self.use_calibration and device_type in self.calibration_factors:
            total *= self.calibration_factors[device_type]

        return total

    def calculate_energy_per_bit(
        self,
        tx_power: float,
        data_rate: float,
        packet_size: int = 1000,
        mode: str = "transmit",
        device_type: str = "obu",
    ) -> float:
        if data_rate <= 0:
            return float("inf")
        tx_time = packet_size / data_rate

        total_power = self.calculate_total_power(tx_power, mode, device_type=device_type)

        total_energy = total_power * tx_time

        energy_per_bit = total_energy / packet_size

        return energy_per_bit

    def calculate_communication_energy(
        self,
        tx_power: float,
        data_rate: float,
        duration: float,
        mode: str = "transmit",
        device_type: str = "obu",
    ) -> dict:
        total_power = self.calculate_total_power(tx_power, mode, device_type=device_type)
        total_energy = total_power * duration
        bits_transmitted = data_rate * duration

        energy_per_bit = total_energy / bits_transmitted if bits_transmitted > 0 else 0

        pa = tx_power / self.params.pa_efficiency
        return {
            "total_power": total_power,
            "total_energy": total_energy,
            "bits_transmitted": bits_transmitted,
            "energy_per_bit": energy_per_bit,
            "pa_power": pa,
            "circuit_power": total_power - pa,
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
        duty_cycle_fraction: float = 1.0,
    ) -> float:
        """
        Extrapolate average per-vehicle CO2 to an annual figure (kg CO2 / vehicle / year).

        Uses scaled annualization:
          (total_co2/n) * (seconds_per_year / simulation_duration_s) * duty_cycle_fraction
        where duty_cycle_fraction captures the fraction of yearly time spent in
        active V2X communication conditions represented by the simulation.
        """
        if (
            num_vehicles <= 0
            or simulation_duration_s <= 0
            or seconds_per_year <= 0
            or duty_cycle_fraction < 0
        ):
            return 0.0
        return self.avg_co2_kg_per_vehicle(total_co2_kg, num_vehicles) * (
            seconds_per_year / simulation_duration_s
        ) * duty_cycle_fraction


class ComprehensiveEnvironmentalMetrics:
    """
    Labeled CO2 breakdown for communication-centric V2X studies.

    **In scope (optional toggles):** direct communication energy → grid CO2;
    optional BS site overhead (cooling + PSU-style losses); optional amortized
    embodied carbon for BS hardware over a nominal lifetime.

    **Out of scope (explicitly excluded):** vehicle propulsion, manufacturing of
    vehicles/OBUs, backbone/core transport, non-modeled user devices. Vehicle
    propulsion is **not** approximated here; state scope in the manuscript.

    When infrastructure and embodied terms are disabled, totals align with
    :class:`EnvironmentalMetrics` for fair comparison across plots.
    """

    def __init__(
        self,
        carbon_intensity_kg_per_kwh: float = 0.5,
        include_infrastructure: bool = True,
        include_embodied_carbon: bool = False,
    ):
        self.carbon_intensity = carbon_intensity_kg_per_kwh
        self.include_infrastructure = include_infrastructure
        self.bs_cooling_overhead = 0.4
        self.power_supply_loss = 0.1
        self.include_embodied_carbon = include_embodied_carbon
        self.bs_embodied_carbon_kg = 2000.0
        self.bs_lifetime_years = 10
        self.bs_operational_hours_per_year = 8760.0

    def calculate_total_co2(
        self,
        communication_energy_j: float,
        num_base_stations: int,
        simulation_duration_s: float,
        include_all_scope: bool = True,
    ) -> Dict[str, Any]:
        co2_breakdown: Dict[str, Any] = {
            "communication_direct_kg": 0.0,
            "infrastructure_overhead_kg": 0.0,
            "embodied_carbon_kg": 0.0,
            "total_kg": 0.0,
            "scope_notes": [],
        }

        comm_energy_kwh = communication_energy_j / 3.6e6
        co2_breakdown["communication_direct_kg"] = (
            comm_energy_kwh * self.carbon_intensity
        )
        co2_breakdown["scope_notes"].append(
            "Direct communication energy (TX chain, baseband proxy)"
        )

        if include_all_scope and self.include_infrastructure:
            infra_overhead = self.bs_cooling_overhead + self.power_supply_loss
            infra_energy_kwh = comm_energy_kwh * infra_overhead
            co2_breakdown["infrastructure_overhead_kg"] = (
                infra_energy_kwh * self.carbon_intensity
            )
            co2_breakdown["scope_notes"].append(
                f"BS infrastructure overhead (cooling + PSU losses; factor ~{1.0 + infra_overhead:.2f} on energy)"
            )

        if include_all_scope and self.include_embodied_carbon:
            sim_fraction_of_year = simulation_duration_s / (
                self.bs_lifetime_years * 365.25 * 24 * 3600
            )
            embodied_per_bs = self.bs_embodied_carbon_kg * sim_fraction_of_year
            co2_breakdown["embodied_carbon_kg"] = embodied_per_bs * num_base_stations
            co2_breakdown["scope_notes"].append(
                f"Embodied carbon amortized over {self.bs_lifetime_years} years"
            )

        co2_breakdown["total_kg"] = (
            co2_breakdown["communication_direct_kg"]
            + co2_breakdown["infrastructure_overhead_kg"]
            + co2_breakdown["embodied_carbon_kg"]
        )
        return co2_breakdown

    def get_scope_statement(self) -> str:
        scope = "CO2 Emissions Scope:\n"
        scope += "INCLUDED:\n"
        scope += "  - Direct communication energy (TX/RX circuits, baseband processing)\n"
        scope += "  - Power amplifier energy consumption\n"
        if self.include_infrastructure:
            scope += "  - BS infrastructure overhead (cooling, PSU losses, PUE-style factor)\n"
        if self.include_embodied_carbon:
            scope += (
                f"  - Embodied carbon of BS hardware (amortized over {self.bs_lifetime_years} years)\n"
            )
        scope += "\nEXCLUDED (out of scope):\n"
        scope += "  - Vehicle propulsion energy for transportation\n"
        scope += "  - Manufacturing emissions of vehicles and OBUs\n"
        scope += "  - Network backbone / transport network energy\n"
        scope += "  - End-user smartphone/tablet energy (non-OBU)\n"
        return scope


def representative_model_epb_predictions(energy_model: EnergyModel) -> Dict[str, float]:
    """EPB at literature-aligned operating points for hardware validation."""
    bs_epb = energy_model.calculate_energy_per_bit(
        20.0,
        100e6,
        packet_size=1000,
        mode="transmit",
        device_type="bs",
    )
    obu_epb = energy_model.calculate_energy_per_bit(
        0.2,
        6e6,
        packet_size=1000,
        mode="transmit",
        device_type="obu",
    )
    return {
        "bs_energy_per_bit": float(bs_epb),
        "obu_energy_per_bit": float(obu_epb),
    }
