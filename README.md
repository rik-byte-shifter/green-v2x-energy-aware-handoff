# Green V2X: Energy-Aware Handoff in Vehicular Networks

This project simulates highway V2X uplink communication and compares five handoff policies:

- `energy_aware`
- `rssi`
- `sinr`
- `load_aware_rssi`
- `naive_nearest`

The simulator reports energy, EPB, throughput, handoff behavior, outage decomposition, and CO2 metrics, and includes validation tooling for literature anchors and duration extrapolation.

---

## What Is Included

- Multi-algorithm comparison with shared PHY/channel assumptions.
- Highway mobility with lane-wise speed ranges and lane switching.
- Weather-aware path-loss and rain attenuation (`clear`, `heavy_rain`, `urban_dense` profile effects).
- Energy model with TX-chain components and optional calibration hook.
- CO2 reporting:
  - simple grid-intensity conversion
  - extended scope breakdown (`co2_breakdown_comprehensive`) + scope statement text.
- Robust experiment suite:
  - multi-seed baseline
  - scaling experiment (20/50/100/200 vehicles)
  - energy-model sensitivity sweeps
  - scenario diversity experiment
  - comprehensive validation study (`--comprehensive-validation`)
- Regression tests in `tests/test_simulation.py`.

---

## Recent Fixes Integrated

The current codebase already includes these fixes:

1. **Network state reset between algorithm runs** (`simulations/simulator.py`)
   - `V2XSimulator.run_algorithm()` calls `_reset_network_state()` so each algorithm starts from clean BS load state.
   - Prevents RSSI/SINR/load-aware baselines from inheriting stale `connected_vehicles` state.

2. **Outage metrics are decomposed** (`simulations/simulator.py`)
   - `outage_probability_percent` = total outage.
   - `coverage_gap_percent` = no BS in coverage.
   - `sinr_outage_percent` = connected but SINR below threshold.
   - Console output and stats now expose this split directly.

3. **Scaling experiment uses dense scenario** (`simulations/config.py`, `main.py`)
   - `run_scaling_experiment()` now uses `SimulationConfig.scaling_scenario()`.
   - Avoids the previous high-vehicle collapse to 0% savings from sparse-coverage gap dominance.

4. **Hardware validation scope is explicit** (`src/utils/hardware_validation.py`)
   - Anchors are tagged with `side` (`obu`/`bs`).
   - `validation_passed` is determined only from OBU in-scope anchors.
   - BS anchors are reported for context (`in_scope=False`) and do not fail validation.

5. **Sparse baseline geometry corrected to avoid single-candidate behavior** (`simulations/config.py`)
   - `sparse_demonstration_scenario()` now uses:
     - `num_base_stations=16`
     - `bs_coverage_radius=400`
     - `shadowing_std_db=10.0`
   - This creates overlapping coverage so RSSI/SINR/load-aware/EA are no longer forced into the same choice each step.

6. **Base station capacity updated for scaling realism** (`src/models/basestation.py`)
   - `BSConfig.max_capacity` default increased from `20` to `100`.
   - Prevents high-load runs (especially 200-vehicle scaling) from artificial association blocking due to too-low toy capacity.

Compatibility keys are also preserved in hardware validation output:
- `mean_absolute_error`
- `max_deviation_percent`
- `mean_absolute_error_obu`

---

## Project Layout

- `main.py`
  - Main experiment runner.
  - Runs baseline multi-seed comparison + plotting + scaling + sensitivity + scenario diversity.
  - Includes Bangladesh grid-intensity sensitivity printout.
- `validation_runner.py`
  - Runs comprehensive validation (hardware + extrapolation + CO2 scope + literature text).
- `weather_sweep.py`
  - Runs multi-weather robustness sweep with uncertainty across seeds.
  - Saves weather-sweep JSON + 95% CI plots in `results/`.
- `debug_metrics.py`
  - Quick local sanity script to compare base-station selection behavior between Energy-Aware and RSSI in a minimal scenario.
- `simulations/config.py`
  - `SimulationConfig` and scenario factories:
    - `sparse_demonstration_scenario()`
    - `scaling_scenario()`
    - `extended_validation_scenario()`
    - `multi_duration_validation()`
    - `sparse_demonstration_scenario_fixed_rssi_tx_sensitivity()`
    - `bangladesh_grid_scenario()`
- `simulations/simulator.py`
  - `V2XSimulator` core simulation logic and per-algorithm metrics.
- `simulations/validator.py`
  - Extrapolation validation and duration stability analysis.
- `src/models/`
  - Vehicle, base station, weather, and energy models.
- `src/algorithms/`
  - Handoff policy implementations.
- `src/utils/hardware_validation.py`
  - Literature anchor comparison + validation scope handling.
- `src/utils/visualization.py`
  - Plot generation.
- `docs/literature_review.py`
  - Literature key index and related-work helper text.
- `results/`
  - Generated plots and JSON outputs.
- `tests/`
  - Regression tests.

---

## Setup

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## How To Run

### 1) Full experiment suite

```powershell
python main.py
```

This runs:
- multi-seed baseline comparison (`SEEDS = [42, 123, 456, 789, 1011]`)
- statistical summary + paired t-tests (RSSI vs Energy-Aware)
- plot generation in `results/`
- scaling experiment
- energy model sensitivity sweep
- scenario diversity experiment
- Bangladesh grid intensity sensitivity printout

Runtime note:
- `python main.py` is a full research suite (many nested multi-seed experiments) and typically takes about **2-3 hours** on a standard laptop CPU.

### 2) Comprehensive validation

```powershell
python main.py --comprehensive-validation
```

or:

```powershell
python validation_runner.py
```

This includes:
- hardware EPB validation against literature anchors
- multi-duration extrapolation validation (`300, 900, 1800, 3600, 7200 s`)
- comprehensive CO2 scope statement and breakdown example
- generated extrapolation plot and validation JSON

### 3) Multi-weather sweep (with uncertainty)

```powershell
python weather_sweep.py
```

Optional useful flags:

```powershell
python weather_sweep.py --seeds 42,123,456,789,1011 --duration 800 --num-vehicles 20 --num-base-stations 9 --area-size 3000
```

This includes:
- weather-profile comparison across configured profiles in `src/models/weather.py`
- mean and 95% confidence intervals across seeds (t-distribution)
- JSON + bar plots for energy saving, handoff reduction, and CO2 saving

---

## Key Output Metrics

Each algorithm produces a `stats` dictionary with fields including:

- Energy and traffic:
  - `total_energy_joules`
  - `total_bits`
  - `avg_energy_per_bit`
  - `avg_tx_power`
  - `avg_throughput_bps`
  - `p5_throughput_bps`
- Handoff and stability:
  - `total_handoffs`
  - `ping_pong_handoffs`
  - `ping_pong_rate_percent`
  - `handoff_delay_total_s`
  - `handoff_delay_per_vehicle_s`
  - `avg_service_availability_percent`
  - `p5_service_availability_percent`
  - `reconnect_events`
  - outage burst statistics
- Outage decomposition:
  - `outage_probability_percent`
  - `coverage_gap_percent`
  - `sinr_outage_percent`
- CO2:
  - `co2_kg`, `co2_grams`
  - `avg_co2_kg_per_vehicle`
  - `co2_kg_per_vehicle_per_year`
  - `co2_breakdown_comprehensive`
  - `co2_scope_statement`

---

## Main Result Artifacts

Typical files written to `results/`:

- Baseline outputs:
  - `simulation_results.json`
  - `bar_comparison.png`
  - `bar_co2_comparison.png`
  - `energy_comparison.png`
  - `energy_per_vehicle_cdf.png`
  - `cumulative_co2.png`
  - `network_topology.png`
  - `tx_power_distribution.png`
  - `bs_load_distribution.png`
  - `sinr_histogram.png`
- Experiment outputs:
  - `scaling_energy_saving_vs_vehicles.png`
  - `scaling_results.json`
  - `energy_model_sensitivity.png`
  - `energy_model_sensitivity.json`
  - `energy_model_justification.md`
  - `scenario_diversity_energy_saving.png`
  - `scenario_diversity_results.json`
- Validation outputs:
  - `extrapolation_validation.png`
  - `comprehensive_validation.json`
- Weather sweep outputs:
  - `weather_sweep.json`
  - `weather_sweep_energy_saving.png`
  - `weather_sweep_handoff_reduction.png`
  - `weather_sweep_co2_saving.png`

---

## Running Tests

Run full tests:

```powershell
python -m pytest tests/ -v
```

Run a single test module:

```powershell
python -m pytest tests/test_simulation.py -v
```

Current suite covers:
- environmental metrics
- weather/path-loss behavior
- QoS and algorithm behavior
- simulator short runs and stats keys
- comprehensive CO2 breakdown consistency
- hardware validator output compatibility

---

## Reproducibility Notes

- Use fixed seeds for comparable runs (`SEEDS`, `SCALING_SEEDS`, `SENSITIVITY_SEEDS`, `SCENARIO_SEEDS` in `main.py`).
- The baseline scenario in `main.py` is `SimulationConfig.sparse_demonstration_scenario()`.
- Current sparse baseline settings are intentionally overlapping (`16` BS, `400 m` radius) to preserve algorithm differentiation.
- The scaling study intentionally switches to `SimulationConfig.scaling_scenario()` for better connectivity at high load.
- Base station default capacity is `BSConfig.max_capacity=100` for realistic high-vehicle scaling behavior.
- Bangladesh-specific CO2 sensitivity uses `SimulationConfig.bangladesh_grid_scenario()` (`0.62 kg CO2/kWh`).

---

## Paper/Reporting Scope Notes

- Hardware validation in this project is **literature-scale sanity checking**, not measured testbed calibration.
- `validation_passed` is based on OBU-side anchors because the simulator models vehicular uplink energy.
- Annualized CO2 is a linear extrapolation from simulated duration via `seconds_per_year`.
- Extended CO2 scope fields are configurable and should be reported with clear in-scope/out-of-scope statements.
