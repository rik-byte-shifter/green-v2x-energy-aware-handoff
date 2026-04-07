"""
Simulation validation: multi-duration stability and extrapolation checks.

Runs the same scenario at several durations and tests whether per-second rates
(energy, CO2, handoffs) have low coefficient of variation—supporting linear
annualization from short runs when rates are stationary enough.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Dict, List

import numpy as np

from simulations.config import SimulationConfig
from simulations.simulator import V2XSimulator


class ExtrapolationValidator:
    """Check whether per-second rates are stable across simulation durations."""

    def __init__(self, base_config: SimulationConfig):
        self.base_config = base_config
        self.results_by_duration: Dict[int, Dict[str, float]] = {}

    def run_multi_duration_study(self, durations: List[int]) -> Dict[str, Any]:
        validation_results: Dict[str, Any] = {
            "durations_tested": list(durations),
            "metrics": {},
            "linearity_analysis": {},
            "extrapolation_valid": True,
        }

        for duration in durations:
            config = replace(self.base_config, duration=duration)
            sim = V2XSimulator(config)
            results = sim.run_comparison()
            ea_stats = results["energy_aware_stats"]
            tot_e = float(ea_stats["total_energy_joules"])
            tot_co2 = float(ea_stats["co2_kg"])
            ho = float(ea_stats["total_handoffs"])
            d = float(duration)
            self.results_by_duration[duration] = {
                "total_energy_j": tot_e,
                "total_co2_kg": tot_co2,
                "total_handoffs": ho,
                "energy_per_second": tot_e / d,
                "co2_per_second": tot_co2 / d,
                "handoffs_per_second": ho / d,
            }

        for metric in ("energy_per_second", "co2_per_second", "handoffs_per_second"):
            values = [self.results_by_duration[d][metric] for d in durations]
            mean_v = float(np.mean(values))
            std_v = float(np.std(values))
            cv = float(std_v / mean_v * 100.0) if mean_v > 0 else 0.0
            stable = cv < 10.0
            validation_results["metrics"][metric] = {
                "values_by_duration": {str(d): self.results_by_duration[d][metric] for d in durations},
                "mean": mean_v,
                "std": std_v,
                "coefficient_of_variation_percent": cv,
                "stable": stable,
            }
            if not stable:
                validation_results["extrapolation_valid"] = False
                validation_results["linearity_analysis"][metric] = (
                    f"High variation (CV={cv:.1f}%) - extrapolation may be unreliable"
                )
            else:
                validation_results["linearity_analysis"][metric] = (
                    f"Stable (CV={cv:.1f}%) - linear extrapolation reasonable for this metric"
                )

        return validation_results

    def generate_validation_plot(self, output_path: str) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.results_by_duration:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ["energy_per_second", "co2_per_second", "handoffs_per_second"]
        titles = ["Energy/s", "CO2/s", "Handoffs/s"]

        for ax, metric, title in zip(axes, metrics, titles):
            durations = sorted(self.results_by_duration.keys())
            values = [self.results_by_duration[d][metric] for d in durations]
            ax.plot(durations, values, "o-", linewidth=2, markersize=8)
            ax.axhline(
                np.mean(values),
                color="r",
                linestyle="--",
                label=f"Mean: {np.mean(values):.4g}",
            )
            ax.set_xlabel("Simulation Duration (s)")
            ax.set_ylabel(title)
            ax.set_title(f"{title} vs Simulation Duration")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
