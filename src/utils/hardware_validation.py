"""
Hardware validation: compare energy model predictions to measured or published hardware.

Compares analytic EPB (J/bit) against literature- and standards-style anchor points
(BS load models, OBU/DSRC-class radios). This is validation against **cited**
power/energy figures—not a substitute for lab traces or hardware-in-the-loop
calibration on your own testbed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

# Relative error vs anchor below this is flagged as within a loose modeling band.
ACCEPTABLE_RELATIVE_ERROR_PERCENT = 50.0
# Above this, validation_passed becomes False (same threshold as existing behavior).
FAIL_RELATIVE_ERROR_PERCENT = 100.0


@dataclass
class HardwareMeasurement:
    """Single anchor: literature or testbed EPB / power operating point."""

    device_type: str
    tx_power_w: float
    circuit_power_w: float
    total_power_w: float
    data_rate_bps: float
    energy_per_bit: float
    reference: str
    conditions: str


class EnergyModelValidator:
    """
    Validate EPB from the analytic model against published hardware-scale numbers.

    Model predictions should expose ``bs_energy_per_bit`` and ``obu_energy_per_bit``
    (J/bit) at comparable operating points to the anchors below.
    """

    def __init__(self) -> None:
        self.measurements: List[HardwareMeasurement] = [
            HardwareMeasurement(
                device_type="5G_BS_macro",
                tx_power_w=20.0,
                circuit_power_w=150.0,
                total_power_w=170.0,
                data_rate_bps=100e6,
                energy_per_bit=1.7e-6,
                reference="3GPP TR 38.840; Björnson & Sanguinetti (2020) OJ-COMS",
                conditions="Loaded BS, 20 MHz bandwidth (illustrative EPB scale)",
            ),
            HardwareMeasurement(
                device_type="LTE_BS_small",
                tx_power_w=5.0,
                circuit_power_w=50.0,
                total_power_w=55.0,
                data_rate_bps=50e6,
                energy_per_bit=1.1e-6,
                reference="Auer et al., IEEE Wireless Commun., 2011",
                conditions="Small cell, moderate load (illustrative)",
            ),
            HardwareMeasurement(
                device_type="OBU_802.11p",
                tx_power_w=0.2,
                circuit_power_w=0.5,
                total_power_w=0.7,
                data_rate_bps=6e6,
                energy_per_bit=1.17e-7,
                reference="IEEE 802.11p / WAVE OBU class (illustrative DSRC scale)",
                conditions="Vehicle OBU, TX mode",
            ),
            # Additional macro-cell style anchor (same ``bs_energy_per_bit`` mapping as above).
            HardwareMeasurement(
                device_type="LTE_BS_macro",
                tx_power_w=40.0,
                circuit_power_w=260.0,
                total_power_w=300.0,
                data_rate_bps=150e6,
                energy_per_bit=2.0e-6,
                reference="Auer et al., IEEE Wireless Commun., 2011 (macro RBS order-of-magnitude)",
                conditions="Macro BS, moderate load (illustrative)",
            ),
        ]

    def validate_model(self, model_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare model EPB keys to `measurements`.

        Expects e.g. ``bs_energy_per_bit``, ``obu_energy_per_bit`` (J/bit).
        """
        results: Dict[str, Any] = {
            "validation_passed": True,
            "comparisons": [],
            "mean_absolute_error": 0.0,
            "max_deviation_percent": 0.0,
        }

        errors: List[float] = []
        for meas in self.measurements:
            if meas.device_type.startswith("5G_BS") or meas.device_type.startswith(
                "LTE_BS"
            ):
                model_epb = model_predictions.get("bs_energy_per_bit")
            elif "OBU" in meas.device_type:
                model_epb = model_predictions.get("obu_energy_per_bit")
            else:
                continue

            if model_epb is None:
                continue

            abs_error = abs(float(model_epb) - meas.energy_per_bit)
            rel_error = abs_error / meas.energy_per_bit * 100.0

            results["comparisons"].append(
                {
                    "device": meas.device_type,
                    "measured_epb": meas.energy_per_bit,
                    "model_epb": float(model_epb),
                    "absolute_error": abs_error,
                    "relative_error_percent": rel_error,
                    "reference": meas.reference,
                    "within_acceptable_range": rel_error
                    < ACCEPTABLE_RELATIVE_ERROR_PERCENT,
                }
            )
            errors.append(rel_error)

            if rel_error > FAIL_RELATIVE_ERROR_PERCENT:
                results["validation_passed"] = False

        if errors:
            results["mean_absolute_error"] = float(np.mean(errors))
            results["max_deviation_percent"] = float(np.max(errors))

        return results

    def get_calibration_factor(self, device_type: str) -> float:
        """Return a multiplier (typically 0.8–1.2) to align model power with anchors."""
        if "BS" in device_type:
            return 1.0
        if "OBU" in device_type:
            return 1.1
        return 1.0
