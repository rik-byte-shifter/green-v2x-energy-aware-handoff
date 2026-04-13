import itertools
import numpy as np
import json
import os
from simulations.config import SimulationConfig
from simulations.simulator import V2XSimulator

def run_sweep():
    # Parameter grid
    param_grid = {
        'bs_coverage_radius': [350, 400, 450],
        'tx_power_max_watts': [0.75, 1.0],
        'sinr_handoff_trigger_db': [1.0, 2.0, 3.0],
        'max_acceptable_epb': [1.5e-7, 2.0e-7],
        'hysteresis': [0.15, 0.20, 0.25],
    }
    seeds = [42, 123, 456]
    results = []

    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())

    print("Starting parameter sweep...")
    for combo in itertools.product(*param_values):
        p = dict(zip(param_keys, combo))
        seed_metrics = []

        for seed in seeds:
            cfg = SimulationConfig.paper_baseline_scenario()
            cfg.seed = seed
            cfg.bs_coverage_radius = p['bs_coverage_radius']
            cfg.tx_power_max_watts = p['tx_power_max_watts']

            sim = V2XSimulator(cfg)
            # Apply algorithm-specific thresholds
            sim.energy_aware_algo.sinr_handoff_trigger_db = p['sinr_handoff_trigger_db']
            sim.energy_aware_algo.max_acceptable_epb = p['max_acceptable_epb']
            sim.energy_aware_algo.hysteresis = p['hysteresis']

            sim.run_comparison()
            ea = sim.results['energy_aware']['stats']
            rssi = sim.results['rssi']['stats']

            energy_saving = (rssi['avg_energy_per_bit'] - ea['avg_energy_per_bit']) / rssi['avg_energy_per_bit'] * 100
            seed_metrics.append({
                'energy_saving_pct': energy_saving,
                'availability_pct': ea['avg_service_availability_percent'],
                'outage_pct': ea['outage_probability_percent'],
                'coverage_gap_pct': ea['coverage_gap_percent'],
                'handoffs': ea['total_handoffs']
            })

        avg = {k: np.mean([m[k] for m in seed_metrics]) for k in seed_metrics[0]}
        avg.update(p)
        results.append(avg)
        print(f"Config: {p} | Energy: {avg['energy_saving_pct']:.1f}% | Avail: {avg['availability_pct']:.1f}%")

    os.makedirs('results', exist_ok=True)
    with open('results/parameter_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to results/parameter_sweep.json")

if __name__ == "__main__":
    run_sweep()
