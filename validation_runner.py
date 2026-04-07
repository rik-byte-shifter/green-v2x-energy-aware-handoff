"""
Comprehensive validation suite: hardware EPB checks, multi-duration stability,
extended CO2 scope text, and literature index.
"""

from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from docs.literature_review import generate_related_work_section, list_all_keys
from simulations.config import SimulationConfig
from simulations.validator import ExtrapolationValidator
from src.models.energy import ComprehensiveEnvironmentalMetrics, EnergyModel, representative_model_epb_predictions
from src.utils.hardware_validation import EnergyModelValidator


def run_comprehensive_validation(
    *,
    durations: list[int] | None = None,
    base_config: SimulationConfig | None = None,
    json_path: str | None = None,
) -> dict:
    """
    Run hardware EPB validation, extrapolation study, CO2 scope example, and print literature.

    Multi-duration runs invoke full ``run_comparison()`` per duration (can be slow).
    If ``durations`` is omitted, uses :meth:`SimulationConfig.multi_duration_validation`.
    """
    print("=" * 70)
    print("COMPREHENSIVE VALIDATION STUDY")
    print("=" * 70)

    base = base_config or SimulationConfig.sparse_demonstration_scenario()
    if durations is None:
        durations = [d for d, _ in SimulationConfig.multi_duration_validation()]

    print("\n1. HARDWARE MODEL VALIDATION")
    print("-" * 70)
    em = EnergyModel(use_calibration=base.use_energy_calibration)
    model_predictions = representative_model_epb_predictions(em)
    validator = EnergyModelValidator()
    validation_results = validator.validate_model(model_predictions)
    print(
        f"Mean relative error vs literature anchors: {validation_results['mean_absolute_error']:.1f}%"
    )
    print(f"Max deviation: {validation_results['max_deviation_percent']:.1f}%")
    print(f"Validation passed: {validation_results['validation_passed']}")
    for comp in validation_results["comparisons"]:
        print(
            f"  {comp['device']}: {comp['relative_error_percent']:.1f}% error "
            f"({comp['reference']})"
        )

    print("\n2. EXTRAPOLATION VALIDATION")
    print("-" * 70)
    dur_list = durations
    extrapolation_validator = ExtrapolationValidator(base)
    extrapolation_results = extrapolation_validator.run_multi_duration_study(dur_list)
    print(f"Extrapolation valid (CV < 10% on rates): {extrapolation_results['extrapolation_valid']}")
    for metric, analysis in extrapolation_results["linearity_analysis"].items():
        print(f"  {metric}: {analysis}")

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "extrapolation_validation.png")
    extrapolation_validator.generate_validation_plot(plot_path)
    print(f"Saved extrapolation plot: {plot_path}")

    print("\n3. COMPREHENSIVE CO2 ACCOUNTING")
    print("-" * 70)
    env_metrics = ComprehensiveEnvironmentalMetrics(
        carbon_intensity_kg_per_kwh=base.carbon_intensity_kg_per_kwh,
        include_infrastructure=base.comprehensive_co2_include_infrastructure,
        include_embodied_carbon=True,
    )
    print(env_metrics.get_scope_statement())
    co2_breakdown = env_metrics.calculate_total_co2(
        communication_energy_j=10000.0,
        num_base_stations=base.num_base_stations,
        simulation_duration_s=3600.0,
        include_all_scope=True,
    )
    print("\nCO2 breakdown example (10 kJ communication energy, 1 hour):")
    print(f"  Direct communication: {co2_breakdown['communication_direct_kg'] * 1000:.2f} g")
    print(f"  Infrastructure: {co2_breakdown['infrastructure_overhead_kg'] * 1000:.2f} g")
    print(f"  Embodied carbon: {co2_breakdown['embodied_carbon_kg'] * 1000:.2f} g")
    print(f"  Total: {co2_breakdown['total_kg'] * 1000:.2f} g")

    print("\n4. LITERATURE REVIEW (draft related-work keys)")
    print("-" * 70)
    print(generate_related_work_section())
    print(f"(Bibliography keys: {len(list_all_keys())} entries in docs/literature_review.py)")

    out = {
        "hardware_validation": validation_results,
        "model_epb_operating_points": model_predictions,
        "extrapolation_validation": extrapolation_results,
        "co2_example_breakdown": co2_breakdown,
        "co2_scope": env_metrics.get_scope_statement(),
    }
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved validation JSON: {json_path}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    return out


if __name__ == "__main__":
    jp = os.path.join(PROJECT_ROOT, "results", "comprehensive_validation.json")
    run_comprehensive_validation(json_path=jp)
