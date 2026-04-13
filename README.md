# Green V2X: Energy-Aware Handoff in Vehicular Networks

Discrete-event style simulation of **highway V2X uplink** with multiple base-station handoff policies. The codebase compares algorithms on energy (including energy-per-bit), throughput, handoffs, outage decomposition, and grid-derived CO2, and ships plotting, multi-seed statistics, validation helpers, and optional sweeps.

---

## Handoff algorithms

| Key | Module | Role |
|-----|--------|------|
| `energy_aware` | `src/algorithms/energy_aware_handoff.py` | Energy- and load-aware target selection |
| `rssi` | `src/algorithms/rssi_handoff.py` | Strongest RSSI |
| `sinr` | `src/algorithms/sinr_handoff.py` | SINR-based |
| `load_aware_rssi` | `src/algorithms/load_aware_rssi_handoff.py` | RSSI with load bias |
| `naive_nearest` | `src/algorithms/naive_nearest_handoff.py` | Geographic nearest BS |

---

## What the simulator models

- **Mobility**: Highway mode with multiple lanes, lane-dependent speeds, and stochastic lane changes (`movement_mode="highway"` in `SimulationConfig`).
- **Radio**: Path loss, shadowing, TX power control, and **weather-dependent** extra loss via profiles in `src/models/weather.py` (`clear`, `light_rain`, `heavy_rain`, `snow`, `fog`, `thunderstorm`, `dust_sand_storm`).
- **Network**: Base stations on a grid, association and capacity (`src/models/basestation.py`), uplink-oriented metrics in `simulations/simulator.py`.
- **Energy / CO2**: TX-chain energy model (`src/models/energy.py`), optional calibration flag, grid **carbon intensity** (kg CO₂/kWh), extended CO2 breakdown fields for reporting (`co2_breakdown_comprehensive`, `co2_scope_statement`).
- **Experiments** (via `main.py`): multi-seed baseline, paired t-tests (RSSI vs energy-aware), scaling across vehicle counts, energy-parameter sensitivity, scenario diversity, Bangladesh grid-intensity spot check, and plots under `results/`.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `main.py` | Full experiment suite: baseline, plots, scaling, sensitivity, scenario diversity |
| `validation_runner.py` | Comprehensive validation (hardware EPB anchors, duration extrapolation, CO2 scope, literature helper text) |
| `weather_sweep.py` | Multi-weather robustness with 95% CIs across seeds; JSON + bar plots |
| `sweep_connectivity_energy.py` | Grid search over EA thresholds and PHY knobs → `results/parameter_sweep.json` (long runtime) |
| `verify_fixes.py` | Fast sanity checks (~1 min): coverage geometry, BS capacity, algorithm differentiation |
| `check_results.py` | Prints numeric stats from `results/simulation_results.json` (debugging) |
| `debug_metrics.py` | Minimal scenario: compare energy-aware vs RSSI BS selection |
| `simulations/config.py` | `SimulationConfig` and scenario factories (`paper_baseline_scenario`, `scaling_scenario`, `extended_validation_scenario`, `multi_duration_validation`, `bangladesh_grid_scenario`, etc.) |
| `simulations/simulator.py` | `V2XSimulator`, per-algorithm runs, comparison metrics, state reset between algorithms |
| `simulations/validator.py` | Extrapolation / duration stability analysis |
| `src/models/` | `vehicle`, `basestation`, `channel`, `weather`, `energy` |
| `src/algorithms/` | Handoff policies (table above) |
| `src/utils/visualization.py` | Standard result plots written to `results/` |
| `src/utils/metrics.py` | Metric helpers |
| `src/utils/hardware_validation.py` | Literature anchor comparison; OBU-scoped pass/fail |
| `docs/literature_review.py` | Literature key index and related-work text helpers |
| `tests/test_simulation.py` | `pytest` regression suite |
| `notebooks/analysis.ipynb` | Optional exploratory analysis |
| `simulations/Untitled-1.ipynb` | Ad-hoc notebook (e.g. custom figures such as Pareto-style plots) |
| `results/` | Generated JSON and PNGs (git may track examples; safe to regenerate) |
| `LICENSE` | MIT |

---

## Setup

From the project root (Windows **PowerShell** example):

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies include `numpy`, `scipy`, `matplotlib`, `pandas`, `plotly`, `seaborn`, `tqdm`, `jupyter`, and `pytest` (see `requirements.txt`).

---

## How to run

### 1) Full experiment suite

```powershell
python main.py
```

This runs:

- Multi-seed baseline comparison (`SEEDS` in `main.py`: `42, 123, 456, 789, 1011`)
- Statistical summary and paired t-tests (total energy and average EPB: energy-aware vs RSSI)
- Plots via `ResultVisualizer.generate_all_plots()` → `results/`
- Scaling study (`scaling_scenario`, vehicle counts 20–200)
- Energy model sensitivity (PA efficiency and TX circuit power grid)
- Scenario diversity experiment
- Short Bangladesh grid-intensity runs (`bangladesh_grid_scenario`, `0.62` kg CO₂/kWh)

**Runtime**: Often on the order of **tens of minutes to a few hours** on a laptop CPU, depending on load.

**CLI**:

```powershell
python main.py --comprehensive-validation
```

delegates to `validation_runner.run_comprehensive_validation()` (same as running `validation_runner.py` directly).

### 2) Comprehensive validation

```powershell
python validation_runner.py
```

or:

```powershell
python main.py --comprehensive-validation
```

Produces artifacts such as `results/comprehensive_validation.json` and `results/extrapolation_validation.png` (paths are set inside `validation_runner.py`).

### 3) Weather sweep (uncertainty across seeds)

```powershell
python weather_sweep.py
```

Useful options:

```powershell
python weather_sweep.py --seeds 42,123,456 --weathers clear,heavy_rain,fog --duration 800 --time-step 1.0 --num-vehicles 20 --num-base-stations 16 --area-size 2800 --results-dir results
```

- Starts from `SimulationConfig.paper_baseline_scenario()` then applies the CLI overrides for duration, vehicles, BS count, and area.
- Default CLI uses `--num-base-stations 9` and `--area-size 3000` unless you override them; for parity with the **paper baseline** geometry, pass `--num-base-stations 16 --area-size 2800` (see below).
- `--results-dir` controls where `weather_sweep.json` and the three bar plots are written (default `results`).

### 4) Energy-aware hyperparameter / connectivity sweep

```powershell
python sweep_connectivity_energy.py
```

Exhaustive Cartesian product over several parameters; writes `results/parameter_sweep.json`. Expect **long** runtimes.

### 5) Quick verification (no full suite)

```powershell
python verify_fixes.py
```

### 6) Tests

```powershell
python -m pytest tests/ -v
```

---

## Key output metrics

Each algorithm’s `stats` include energy and traffic (`total_energy_joules`, `avg_energy_per_bit`, `avg_throughput_bps`, `p5_throughput_bps`, …), handoff behavior (`total_handoffs`, `ping_pong_rate_percent`, `handoff_delay_*`, …), service quality (`avg_service_availability_percent`, …), **outage decomposition** (`outage_probability_percent`, `coverage_gap_percent`, `sinr_outage_percent`), and CO2 fields (`co2_kg`, `co2_breakdown_comprehensive`, `co2_scope_statement`, …).

`run_comparison()` also reports relative improvements vs RSSI, SINR, load-aware RSSI, and naive nearest (energy, CO2, handoffs, etc.).

---

## Typical artifacts under `results/`

| Group | Files |
|-------|--------|
| Baseline (from `main.py` + visualizer) | `simulation_results.json`, `bar_comparison.png`, `bar_co2_comparison.png`, `energy_comparison.png`, `energy_per_vehicle_cdf.png`, `cumulative_co2.png`, `network_topology.png`, `tx_power_distribution.png`, `bs_load_distribution.png`, `sinr_histogram.png` |
| Scaling / sensitivity / diversity | `scaling_energy_saving_vs_vehicles.png`, `scaling_results.json`, `energy_model_sensitivity.png`, `energy_model_sensitivity.json`, `energy_model_justification.md`, `scenario_diversity_energy_saving.png`, `scenario_diversity_results.json` |
| Validation | `extrapolation_validation.png`, `comprehensive_validation.json` |
| Weather sweep | `weather_sweep.json`, `weather_sweep_energy_saving.png`, `weather_sweep_handoff_reduction.png`, `weather_sweep_co2_saving.png` |
| Optional sweep | `parameter_sweep.json` (from `sweep_connectivity_energy.py`) |

Custom figures (e.g. Pareto-style plots) may be produced from notebooks rather than `main.py`.

---

## Reproducibility and design notes

- **Seeds** are centralized in `main.py` (`SEEDS`, `SCALING_SEEDS`, `SENSITIVITY_SEEDS`, `SCENARIO_SEEDS`).
- **Primary baseline** for `main.py` is `SimulationConfig.paper_baseline_scenario()`:
  - `num_base_stations=16` (4×4 grid), `area_size=2800`, `bs_coverage_radius=450`, `weather_profile="clear"`, `shadowing_std_db=10.0`, with overlapping coverage so RSSI / SINR / load-aware / energy-aware policies can diverge meaningfully.
- **Scaling experiment** uses `SimulationConfig.scaling_scenario()` (e.g. larger coverage radius) so high vehicle counts are not dominated by coverage gaps.
- **BS capacity**: default `BSConfig.max_capacity=100` avoids toy capacity blocking at 200 vehicles.
- **Simulator**: `V2XSimulator.run_algorithm()` resets network state between algorithms so BS load is not carried across runs.
- **Hardware validation** is literature-scale EPB sanity checking, not bench calibration; `validation_passed` follows **OBU-side** anchors in `hardware_validation.py`.

---

## License

This project is licensed under the MIT License; see `LICENSE`.
