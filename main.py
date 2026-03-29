"""
Green V2X: Energy-Aware Handoff in Vehicular Networks
Main execution script
"""

import os
import sys

from simulations.simulator import V2XSimulator
from simulations.config import SimulationConfig
from src.utils.visualization import ResultVisualizer


def main():
    print("=" * 70)
    print("GREEN V2X: ENERGY-AWARE HANDOFF IN VEHICULAR NETWORKS")
    print("=" * 70)

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    config = SimulationConfig(
        num_vehicles=20,
        num_base_stations=9,
        duration=500,
        area_size=2000,
        bs_coverage_radius=300,
        vehicle_speed_min=10,
        vehicle_speed_max=30,
        seed=42
    )

    print(f"\nSimulation Configuration:")
    print(f"  Vehicles: {config.num_vehicles}")
    print(f"  Base Stations: {config.num_base_stations}")
    print(f"  Duration: {config.duration}s")
    print(f"  Area: {config.area_size}m x {config.area_size}m")
    print(f"  Speed Range: {config.vehicle_speed_min}-{config.vehicle_speed_max} m/s")

    simulator = V2XSimulator(config)

    comparison = simulator.run_comparison()

    results_path = os.path.join(root, 'results', 'simulation_results.json')
    simulator.save_results(results_path)

    visualizer = ResultVisualizer(simulator.results, results_dir=os.path.join(root, 'results'))
    visualizer.generate_all_plots(
        vehicles=simulator.vehicles,
        base_stations=simulator.base_stations
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    ea_stats = comparison['energy_aware_stats']
    rssi_stats = comparison['rssi_stats']

    print(f"\nEnergy-Aware Algorithm:")
    print(f"  Total Energy: {ea_stats['total_energy_joules']:.2f} J")
    print(f"  Energy-per-Bit: {ea_stats['avg_energy_per_bit']*1e6:.4f} uJ/bit")
    print(f"  Total Handoffs: {ea_stats['total_handoffs']}")
    print(f"  Avg TX Power: {ea_stats['avg_tx_power']*1000:.2f} mW")

    print(f"\nRSSI-Based Algorithm:")
    print(f"  Total Energy: {rssi_stats['total_energy_joules']:.2f} J")
    print(f"  Energy-per-Bit: {rssi_stats['avg_energy_per_bit']*1e6:.4f} uJ/bit")
    print(f"  Total Handoffs: {rssi_stats['total_handoffs']}")
    print(f"  Avg TX Power: {rssi_stats['avg_tx_power']*1000:.2f} mW")

    print(f"\nIMPROVEMENTS:")
    print(f"  Energy Saving: {comparison['energy_saving_percent']:.2f}%")
    print(f"  Handoff Reduction: {comparison['handoff_reduction_percent']:.2f}%")

    print("\n" + "=" * 70)
    print("Simulation completed successfully!")
    print("Check 'results/' folder for plots and data")
    print("=" * 70)

    return comparison


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
