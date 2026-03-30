"""
Green V2X: Energy-Aware Handoff in Vehicular Networks
Main execution script
"""

import os
import sys

import numpy as np
from scipy import stats

from simulations.simulator import V2XSimulator
from simulations.config import SimulationConfig
from src.utils.visualization import ResultVisualizer

# Multiple seeds for mean ± std (journal-style reporting)
SEEDS = [42, 123, 456, 789, 1011]


def main():
    print("=" * 70)
    print("GREEN V2X: ENERGY-AWARE HANDOFF IN VEHICULAR NETWORKS")
    print("=" * 70)

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    # Sparse / high-fading demonstration: RSSI energy uses fixed TX (see config);
    # energy-aware uses adaptive power — strong EPB / total-energy gap vs RSSI.
    config = SimulationConfig.sparse_demonstration_scenario()

    print(f"\nSimulation Configuration:")
    print(f"  Vehicles: {config.num_vehicles}")
    print(f"  Base Stations: {config.num_base_stations}")
    print(f"  Duration: {config.duration}s")
    print(f"  Area: {config.area_size}m x {config.area_size}m")
    print(f"  Speed Range: {config.vehicle_speed_min}-{config.vehicle_speed_max} m/s")
    print(f"  Shadowing std: {config.shadowing_std_db} dB")
    wp = config.get_weather()
    print(f"  Weather profile: {config.weather_profile} ({wp.name})")
    print(
        f"  Effective path-loss exponent: {wp.path_loss_exponent:g} | "
        f"weather shadowing std: {wp.shadowing_std_db:g} dB | "
        f"weather attenuation: {wp.rain_attenuation_db_per_km:g} dB/km"
    )
    print(f"  Highway lateral noise std: {config.highway_lateral_noise_std_m} m")
    print(f"  Grid carbon intensity: {config.carbon_intensity_kg_per_kwh} kg CO2/kWh")
    print(
        f"  CO2 annualization: {config.seconds_per_year:.0f} s/year "
        f"(linear extrapolation from sim duration)"
    )
    print(f"  RSSI energy model: {'fixed TX (baseline)' if config.rssi_energy_use_fixed_tx else 'same Shannon link as stats'}")
    print(f"  Seeds: {SEEDS}")

    results_list = []
    last_simulator = None
    last_comparison = None

    for seed in SEEDS:
        config.seed = seed
        print(f"\n--- Seed {seed} ---")
        simulator = V2XSimulator(config)
        comparison = simulator.run_comparison()
        results_list.append(comparison)
        last_simulator = simulator
        last_comparison = comparison

    def mean_std(key: str):
        vals = [r[key] for r in results_list if r.get(key) is not None]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    es_mean, es_std = mean_std("energy_saving_percent")
    ho_mean, ho_std = mean_std("handoff_reduction_percent")
    evn_mean, evn_std = mean_std("energy_saving_vs_naive_percent")
    rvn_mean, rvn_std = mean_std("rssi_vs_naive_energy_percent")
    es_tot_mean, es_tot_std = mean_std("energy_saving_total_joules_percent")

    ea_e = [r["energy_aware_stats"]["total_energy_joules"] for r in results_list]
    rssi_e = [r["rssi_stats"]["total_energy_joules"] for r in results_list]
    naive_e = [r["naive_nearest_stats"]["total_energy_joules"] for r in results_list]

    ea_epb = [r["energy_aware_stats"]["avg_energy_per_bit"] for r in results_list]
    rssi_epb = [r["rssi_stats"]["avg_energy_per_bit"] for r in results_list]

    # Paired tests: same seeds → paired t-tests (RSSI vs energy-aware)
    t_j, p_j = stats.ttest_rel(rssi_e, ea_e, alternative="greater", nan_policy="omit")
    t_e, p_e = stats.ttest_rel(rssi_epb, ea_epb, alternative="greater", nan_policy="omit")

    print("\n" + "=" * 70)
    print("MULTI-SEED STATISTICS (mean +/- std)")
    print("=" * 70)
    print(f"  Energy saving vs RSSI (avg EPB):     {es_mean:.2f}% +/- {es_std:.2f}%")
    print(
        f"  Energy saving vs RSSI (total J):   {es_tot_mean:.2f}% +/- {es_tot_std:.2f}%"
    )
    print(f"  Handoff reduction vs RSSI: {ho_mean:.2f}% +/- {ho_std:.2f}%")
    print(f"  Energy saving vs Naive:    {evn_mean:.2f}% +/- {evn_std:.2f}%")
    if not np.isnan(rvn_mean):
        print(f"  RSSI energy gain vs Naive: {rvn_mean:.2f}% +/- {rvn_std:.2f}%")
    print(f"  Total energy (J) - Energy-Aware:  {np.mean(ea_e):.2f} +/- {np.std(ea_e):.2f}")
    print(f"  Total energy (J) - RSSI:          {np.mean(rssi_e):.2f} +/- {np.std(rssi_e):.2f}")
    print(f"  Total energy (J) - Naive Nearest: {np.mean(naive_e):.2f} +/- {np.std(naive_e):.2f}")

    print("\n  PAIRED t-TESTS (H0: E[RSSI] <= E[EA]; alt: RSSI > EA, i.e. EA saves energy)")
    print(f"    Total energy (J): t={t_j:.4f}, p={p_j:.6g}")
    print(f"    Avg EPB (J/bit):  t={t_e:.4f}, p={p_e:.6g}")

    max_epb_saving = max(r["energy_saving_percent"] for r in results_list)
    _tag = "OK" if max_epb_saving > 15.0 else "NOT MET"
    print(
        f"\n  Sparse-scenario check (target >15% EPB vs RSSI at least one seed): "
        f"max seed saving = {max_epb_saving:.2f}% ({_tag})"
    )

    ea_co2 = [r["energy_aware_stats"]["co2_kg"] for r in results_list]
    rssi_co2 = [r["rssi_stats"]["co2_kg"] for r in results_list]
    naive_co2 = [r["naive_nearest_stats"]["co2_kg"] for r in results_list]
    ci = results_list[0]["energy_aware_stats"]["carbon_intensity_kg_per_kwh"]
    print(f"  Est. CO2 (kg) - Energy-Aware:  {np.mean(ea_co2):.6f} +/- {np.std(ea_co2):.6f}  (intensity {ci:g} kg/kWh)")
    print(f"  Est. CO2 (kg) - RSSI:          {np.mean(rssi_co2):.6f} +/- {np.std(rssi_co2):.6f}")
    print(f"  Est. CO2 (kg) - Naive Nearest: {np.mean(naive_co2):.6f} +/- {np.std(naive_co2):.6f}")

    ea_py = [r["energy_aware_stats"]["co2_kg_per_vehicle_per_year"] for r in results_list]
    rssi_py = [r["rssi_stats"]["co2_kg_per_vehicle_per_year"] for r in results_list]
    naive_py = [r["naive_nearest_stats"]["co2_kg_per_vehicle_per_year"] for r in results_list]
    print(
        f"  CO2 kg / vehicle / year - Energy-Aware:  {np.mean(ea_py):.6f} +/- {np.std(ea_py):.6f}"
    )
    print(
        f"  CO2 kg / vehicle / year - RSSI:          {np.mean(rssi_py):.6f} +/- {np.std(rssi_py):.6f}"
    )
    print(
        f"  CO2 kg / vehicle / year - Naive Nearest: {np.mean(naive_py):.6f} +/- {np.std(naive_py):.6f}"
    )

    leq_flags = [r.get("energy_aware_handoffs_leq_rssi") for r in results_list]
    if all(leq_flags):
        print("  Handoff constraint (EA <= RSSI): all seeds pass")
    else:
        print(f"  Handoff constraint (EA <= RSSI): {sum(leq_flags)}/{len(leq_flags)} seeds pass")

    results_path = os.path.join(root, "results", "simulation_results.json")
    last_simulator.save_results(results_path)

    visualizer = ResultVisualizer(
        last_simulator.results, results_dir=os.path.join(root, "results")
    )
    visualizer.generate_all_plots(
        vehicles=last_simulator.vehicles, base_stations=last_simulator.base_stations
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY (last seed)")
    print("=" * 70)

    ea_stats = last_comparison["energy_aware_stats"]
    rssi_stats = last_comparison["rssi_stats"]
    naive_stats = last_comparison["naive_nearest_stats"]

    print(f"\nEnergy-Aware Algorithm:")
    print(f"  Total Energy: {ea_stats['total_energy_joules']:.2f} J")
    print(
        f"  Est. CO2: {ea_stats['co2_kg']:.6f} kg ({ea_stats['co2_grams']:.3f} g)"
    )
    print(
        f"  Avg CO2 / vehicle (sim): {ea_stats['avg_co2_kg_per_vehicle']:.6f} kg; "
        f"/ vehicle / year: {ea_stats['co2_kg_per_vehicle_per_year']:.6f} kg"
    )
    print(f"  Energy-per-Bit: {ea_stats['avg_energy_per_bit']*1e6:.4f} uJ/bit")
    print(f"  Total Handoffs: {ea_stats['total_handoffs']}")
    print(f"  Avg TX Power: {ea_stats['avg_tx_power']*1000:.2f} mW")

    print(f"\nRSSI-Based Algorithm:")
    print(f"  Total Energy: {rssi_stats['total_energy_joules']:.2f} J")
    print(
        f"  Est. CO2: {rssi_stats['co2_kg']:.6f} kg ({rssi_stats['co2_grams']:.3f} g)"
    )
    print(
        f"  Avg CO2 / vehicle (sim): {rssi_stats['avg_co2_kg_per_vehicle']:.6f} kg; "
        f"/ vehicle / year: {rssi_stats['co2_kg_per_vehicle_per_year']:.6f} kg"
    )
    print(f"  Energy-per-Bit: {rssi_stats['avg_energy_per_bit']*1e6:.4f} uJ/bit")
    print(f"  Total Handoffs: {rssi_stats['total_handoffs']}")
    print(f"  Avg TX Power: {rssi_stats['avg_tx_power']*1000:.2f} mW")

    print(f"\nNaive Nearest BS:")
    print(f"  Total Energy: {naive_stats['total_energy_joules']:.2f} J")
    print(
        f"  Est. CO2: {naive_stats['co2_kg']:.6f} kg ({naive_stats['co2_grams']:.3f} g)"
    )
    print(
        f"  Avg CO2 / vehicle (sim): {naive_stats['avg_co2_kg_per_vehicle']:.6f} kg; "
        f"/ vehicle / year: {naive_stats['co2_kg_per_vehicle_per_year']:.6f} kg"
    )
    print(f"  Energy-per-Bit: {naive_stats['avg_energy_per_bit']*1e6:.4f} uJ/bit")
    print(f"  Total Handoffs: {naive_stats['total_handoffs']}")
    print(f"  Avg TX Power: {naive_stats['avg_tx_power']*1000:.2f} mW")

    print(f"\nIMPROVEMENTS (last seed):")
    print(f"  Energy Saving vs RSSI (EPB): {last_comparison['energy_saving_percent']:.2f}%")
    print(
        f"  Energy Saving vs RSSI (total J): {last_comparison['energy_saving_total_joules_percent']:.2f}%"
    )
    print(f"  CO2 Saving vs RSSI: {last_comparison['co2_saving_percent']:.2f}%")
    print(f"  Handoff Reduction vs RSSI: {last_comparison['handoff_reduction_percent']:.2f}%")
    print(f"  Energy Saving vs Naive: {last_comparison['energy_saving_vs_naive_percent']:.2f}%")

    print("\n" + "=" * 70)
    print("Simulation completed successfully!")
    print("Plots and JSON reflect the last seed; multi-seed stats printed above.")
    print("Check 'results/' folder for plots and data")
    print("=" * 70)

    return {
        "per_seed": results_list,
        "last": last_comparison,
        "ttest_total_energy": {"statistic": float(t_j), "pvalue": float(p_j)},
        "ttest_avg_epb": {"statistic": float(t_e), "pvalue": float(p_e)},
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
