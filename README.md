## Green V2X: Energy-Aware Handoff in Vehicular Networks

Python framework for simulating vehicular V2X networks and comparing classical **RSSI-based** handoff with an **energy-aware** handoff algorithm.  
The energy-aware policy explicitly minimizes **energy-per-bit (EPB)** with a **load penalty**, while respecting QoS (throughput, outage, service availability) and controlling ping‑pong handoffs.

---

## 1. Features at a Glance

- **Multiple handoff algorithms**
  - **Energy-aware** (EPB × load minimization with hysteresis and time‑to‑trigger)
  - **RSSI** baseline (strongest signal, PHY‑fair adaptive power)
  - **SINR-based** handoff
  - **Load-aware RSSI**
  - **Naive nearest base station**
- **Rich channel and mobility modeling**
  - Grid of base stations with coverage radius and shadowing
  - Highway and non‑highway mobility, lane structure, and rare lane changes
  - Weather profiles (clear / heavy rain / urban‑dense) with path‑loss and rain attenuation
- **Energy and CO₂ accounting**
  - Full TX‑chain energy model (PA, RF circuits, baseband, cooling)
  - Optional **literature-aligned calibration** on OBU/BS-style factors (see Hardware validation)
  - **Simple grid CO₂** (kg CO₂ per kWh) for primary reporting: `co2_kg`, per‑vehicle, annualized figures
  - **Extended CO₂ scope** (optional breakdown): direct comms energy, infrastructure overhead factors, optional amortized embodied BS carbon — with explicit **in‑scope / out‑of‑scope** text for papers
- **Hardware / literature validation**
  - Compare EPB at fixed operating points to published anchor values (`EnergyModelValidator`)
  - Representative model predictions via `representative_model_epb_predictions()` in `src/models/energy.py`
- **Duration / extrapolation validation**
  - Run simulations at multiple durations and check stability of per‑second rates (energy, CO₂, handoffs) using coefficient of variation
  - Helpers: `SimulationConfig.extended_validation_scenario()`, `SimulationConfig.multi_duration_validation()`
- **Literature index for writing**
  - `docs/literature_review.py`: ~24 structured BibTeX-style keys, grouped by topic (green RAN, ICT footprint, V2X, handover/mobility energy, grid factors, channel/PHY), plus `generate_related_work_section()` with draft `\cite{...}` paragraphs
- **Experiment suite**
  - Baseline multi‑seed comparison (robust statistics)
  - **Scalability** vs number of vehicles
  - **Energy‑model sensitivity** (PA efficiency, TX circuit power)
  - **Scenario diversity** (clear / heavy rain / urban dense)
  - **Comprehensive validation** (hardware EPB check, multi‑duration study, CO₂ scope example, literature printout)
- **Automated plots & JSON outputs**
  - Plots for energy, CO₂, SINR, BS load, TX power distribution, CDFs, and diversity/scaling
  - Machine‑readable `results/simulation_results.json` and additional experiment JSON files

---

## 2. Project Structure

- `main.py`
  - Entry point. Runs a sparse/high‑fading demonstration scenario using multiple random seeds.
  - **CLI:** `python main.py` (default) or `python main.py --comprehensive-validation` (full validation suite; see §8).
  - Performs:
    - Multi‑seed comparison of all algorithms
    - Statistical summaries (mean ± std, paired t‑tests)
    - Plot generation and saving of JSON results
    - Scalability, energy‑model sensitivity, and scenario‑diversity experiments
  - Uses `dataclasses.replace()` to clone `SimulationConfig` (so new config fields stay in sync automatically).
- `validation_runner.py`
  - `run_comprehensive_validation()`: hardware EPB validation, extrapolation study, CO₂ breakdown example, literature section output.
  - Run directly: `python validation_runner.py` → also writes `results/comprehensive_validation.json`.
- `simulations/`
  - `simulator.py` — `V2XSimulator`: network setup, vehicles, time steps, handoffs, metrics, `run_comparison()`, `save_results()`.
    - Uses a **single shared** `EnergyModel` for all algorithms that need EPB (energy‑aware, RSSI, SINR, load‑aware RSSI).
    - Stats include **`co2_breakdown_comprehensive`** and **`co2_scope_statement`** (extended scope) alongside legacy `co2_kg`.
  - `config.py` — `SimulationConfig` (network, mobility, PHY, handoff, energy, environment).
  - `validator.py` — `ExtrapolationValidator`: multi‑duration runs, CV analysis, `results/extrapolation_validation.png`.
- `src/models/`
  - `vehicle.py` — mobility state, energy updates (`device_type` for TX chain, default `obu`).
  - `basestation.py` — geometry, link budget, capacity.
  - `energy.py` — `EnergyParams`, `EnergyModel` (optional calibration), `EnvironmentalMetrics` (simple J→CO₂), `ComprehensiveEnvironmentalMetrics` (extended CO₂ + scope text), `representative_model_epb_predictions()`.
- `src/utils/`
  - `hardware_validation.py` — `HardwareMeasurement`, `EnergyModelValidator` (literature anchors vs model EPB).
  - `visualization.py` — plots (energy, CO₂, CDFs, SINR, load, TX power, topology, etc.).
  - `metrics.py` — formatting helpers.
- `docs/`
  - `literature_review.py` — `LITERATURE_REFERENCES` (~24 entries), `generate_related_work_section()`, `list_all_keys()`.
  - Intended as a **source index** for BibTeX / related work; not auto‑synced to `.bib` files.
- `src/algorithms/`
  - `energy_aware_handoff.py`, `rssi_handoff.py`, `sinr_handoff.py`, `load_aware_rssi_handoff.py`, `naive_nearest_handoff.py`
- `results/`
  - `simulation_results.json` — full per‑algorithm stats for the last seed, plus config.
  - Standard plots: `bar_comparison.png`, `bar_co2_comparison.png`, `energy_comparison.png`, `energy_per_vehicle_cdf.png`, `cumulative_co2.png`, `network_topology.png`, etc.
  - Experiments: `scaling_*`, `energy_model_sensitivity.*`, `scenario_diversity_*`, `energy_model_justification.md`
  - Mechanism / distribution plots: `bs_load_distribution.png`, `tx_power_distribution.png`, `sinr_histogram.png`
  - **Validation outputs** (when you run comprehensive validation): `extrapolation_validation.png`, `comprehensive_validation.json`
- `tests/`
  - `test_simulation.py` — regression tests (simulator, metrics, environmental CO₂, hardware validator, comprehensive CO₂ stats keys).
- `notebooks/analysis.ipynb`
  - Optional notebook for ad‑hoc exploration of JSON results.

---

## 3. Installation & Setup

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

The code targets a standard scientific Python stack (NumPy, SciPy, Matplotlib, tqdm, pytest, etc.).  
Matplotlib uses the non‑interactive `Agg` backend so that all plots can be rendered in headless environments.

---

## 4. Running the Simulation Suite

### 4.1 Default full experiment (`main.py`)

From the project root (virtual environment activated):

```powershell
python main.py
```

`main.py` will:

- Instantiate the default **sparse, high‑fading demonstration scenario** via `SimulationConfig.sparse_demonstration_scenario()`.
- Loop over multiple seeds (`SEEDS`) and for each seed:
  - Create a `V2XSimulator`.
  - Run a full comparison of all handoff algorithms.
  - Collect per‑seed metrics (energy, CO₂, throughput, outage, handoffs, etc.).
- Compute **multi‑seed statistics** (mean ± std) for energy savings, CO₂, handoffs, throughput, availability, etc.
- Run **paired t‑tests** on total energy and average EPB (RSSI vs energy‑aware).
- Save `results/simulation_results.json`, generate plots via `ResultVisualizer`, and run scaling / sensitivity / scenario‑diversity experiments.

All output is written into the `results/` directory, which is created automatically if needed.

### 4.2 Comprehensive validation (reviewer-oriented tooling)

```powershell
python main.py --comprehensive-validation
```

or:

```powershell
python validation_runner.py
```

This runs (by default) multi‑duration simulations at **300, 900, 1800, 3600, and 7200 s** (from `SimulationConfig.multi_duration_validation()`), compares model EPB to literature anchors, prints extended CO₂ scope and an example breakdown, prints a draft related‑work paragraph, and saves:

- `results/extrapolation_validation.png`
- `results/comprehensive_validation.json` (when using `validation_runner.py`)

You can call `run_comprehensive_validation(...)` from Python with a smaller `base_config` or shorter `durations` for quick smoke tests.

---

## 5. Experiments Implemented in `main.py`

### 5.1 Baseline Multi‑Seed Comparison

- Seeds: `SEEDS = [42, 123, 456, 789, 1011]`.
- For each seed: `V2XSimulator.run_comparison()` — energy‑aware, RSSI, SINR, load‑aware RSSI, naive nearest.
- Aggregated metrics: EPB and total‑J savings, CO₂ savings, handoffs, ping‑pong, throughput (mean, p5), service availability, outage statistics, paired t‑tests.

### 5.2 Scalability vs Number of Vehicles

- `run_scaling_experiment(root, base_config)` — vehicle counts `[20, 50, 100, 200]`, seeds `SCALING_SEEDS`.
- Outputs: `results/scaling_energy_saving_vs_vehicles.png`, `scaling_results.json`.

### 5.3 Energy Model Sensitivity

- `run_energy_model_sensitivity(root, base_config)` — sweeps PA efficiency and TX circuit power.
- Outputs: `energy_model_sensitivity.png`, `energy_model_sensitivity.json`, `energy_model_justification.md`.

### 5.4 Scenario Diversity

- `run_scenario_diversity_experiment(root, base_config)` — clear weather, heavy rain, urban dense overrides.
- Outputs: `scenario_diversity_energy_saving.png`, `scenario_diversity_results.json`.

---

## 6. Energy, Calibration & CO₂ Metrics

### 6.1 Per‑time‑step energy (vehicular uplink)

- Channel metrics per link (distance, TX power, SINR, rate).
- `EnergyModel.calculate_total_power(tx_power, mode='transmit', device_type='obu')` and `calculate_energy_per_bit(..., device_type='obu')` for vehicle-side accounting.
- Handoffs: fixed `handoff_energy_joules` + transmit energy during `handoff_delay_s`.

### 6.2 Optional calibration (literature alignment)

- `SimulationConfig.use_energy_calibration` (default `True`) enables factors loaded from `EnergyModelValidator.get_calibration_factor()` (e.g. OBU factor ≈ 1.1).
- Set to `False` to disable scaling for ablation or replication of older runs.

### 6.3 Simple CO₂ reporting (primary results)

- `EnvironmentalMetrics.energy_to_co2()` using `carbon_intensity_kg_per_kwh`.
- Stats: `co2_kg`, `co2_grams`, `avg_co2_kg_per_vehicle`, `co2_kg_per_vehicle_per_year` (linear extrapolation via `seconds_per_year`).

### 6.4 Extended CO₂ scope (documentation + optional breakdown)

- `ComprehensiveEnvironmentalMetrics` in `src/models/energy.py`:
  - **Included (when enabled):** direct communication energy; optional infrastructure overhead (cooling + PSU-style factors); optional amortized embodied BS carbon.
  - **Excluded:** propulsion, manufacturing of vehicles/OBUs, backbone transport, smartphone/tablet (non‑OBU) — see `get_scope_statement()`.
- `SimulationConfig`:
  - `comprehensive_co2_include_infrastructure` (default `True`)
  - `comprehensive_co2_include_embodied` (default `False` — embodied can dominate numerically; enable only when you want that breakdown in the paper)
- Each algorithm run’s `stats` include:
  - `co2_breakdown_comprehensive` — dict with `communication_direct_kg`, `infrastructure_overhead_kg`, `embodied_carbon_kg`, `total_kg`, `scope_notes`
  - `co2_scope_statement` — multi-line scope text for copy‑paste into manuscripts

Console logs often use `uJ/bit` for readability; plots may label axes in microjoules per bit.

---

## 7. Hardware & Duration Validation (Conceptual)

- **Hardware validation** (`src/utils/hardware_validation.py`) compares the analytic model’s EPB at fixed operating points to **published anchor rows** (macro/small‑cell BS and OBU/DSRC‑class examples, with relative‑error thresholds in code). It supports **literature‑scale sanity checks**, not a substitute for **measured** OBU/BS traces or a full testbed calibration—state that distinction in sustainability‑oriented venues.
- **Duration validation** (`simulations/validator.py`, class `ExtrapolationValidator`): run the **same** scenario at multiple durations and check whether **per‑second** energy, CO₂, and handoff rates have low **coefficient of variation** (CV under 10% in the default criterion). This supports **linear annualization** arguments for stationary‑enough rates; stochastic runs may occasionally fail strict CV checks—report seeds and scenarios.
- **Config helpers:** `SimulationConfig.extended_validation_scenario()` (e.g. 1 h duration for long runs) and `SimulationConfig.multi_duration_validation()` (labels + duration list used as the default sweep in `run_comprehensive_validation()`).

---

## 8. Literature References (`docs/literature_review.py`)

- Structured dictionary `LITERATURE_REFERENCES` with ~**24** keys across: green wireless networks, ICT footprint (including critical takes such as `freitag2021climate`), V2X, energy‑aware mobility/handover, carbon intensity, channel/PHY.
- `generate_related_work_section()` — draft Markdown with `\cite{...}`-style keys for LaTeX.
- `list_all_keys()` — flat list for bibliography checks.
- **You still need** to add matching BibTeX entries to your `.bib` file and cite in the paper.

---

## 9. Algorithms (Conceptual Overview)

- **Energy‑Aware Handoff** — EPB × load metric, hysteresis, time‑to‑trigger, ping‑pong window.
- **RSSI Baseline** — strongest RSSI; adaptive TX per BS (PHY‑fair vs energy‑aware).
- **SINR‑Based Handoff** — SINR-driven handoffs under the same PHY.
- **Load‑Aware RSSI** — RSSI minus load penalty.
- **Naive Nearest BS** — geometry only.

`V2XSimulator.run_comparison()` aggregates all algorithms and derives relative improvements.

---

## 10. Running Tests

```powershell
python -m pytest tests\ -v
```

or:

```powershell
python -m pytest tests\test_simulation.py -v
```

Tests cover simulator execution, environmental metrics, comprehensive CO₂ math, hardware validator hooks, and presence of extended CO₂ fields in stats.

---

## 11. Reproducing and Extending Experiments

- **Reproduce figures:** run `python main.py` and inspect `results/`.
- **Adjust scenarios:** edit `SimulationConfig` (factories in `config.py` or custom instances). Notable helpers:
  - `sparse_demonstration_scenario()` — default demo
  - `extended_validation_scenario()` — long (1 h) run for stability experiments
  - `multi_duration_validation()` — list of `(duration_s, label)` pairs
- **Tuning:** `area_size`, `num_base_stations`, `bs_coverage_radius`, speeds, weather, shadowing, `EnergyParams`, `use_energy_calibration`, CO₂ scope flags.
- **New algorithms:** add under `src/algorithms/` and wire into `V2XSimulator` like existing policies.

If average TX power sits near the **1 mW floor**, link‑energy differences between algorithms will be small. Increase path‑loss stress (larger area, smaller coverage, harsher weather) or tune `EnergyParams` to amplify differences.

---

## 12. Limitations (for the paper)

The code encodes **scope strings** and configurable CO₂ breakdowns; the **research paper** should still state explicitly, for example:

- Simulation is **abstract** (grid BSs, simplified load, not a full 3GPP stack).
- **Highway** (or chosen) mobility only unless you extend scenarios.
- **Hardware validation** is **literature benchmarking**, not measured OBU/BS traces.
- **Annualized CO₂** uses **linear extrapolation** from the simulated interval; real duty cycles differ.
- **Extended CO₂** uses **illustrative** overhead and embodied factors when enabled — always label which scenario is reported in each figure/table.
- **`--comprehensive-validation`** runs a **full** `run_comparison()` at **each** duration (default: 300–7200 s); wall‑clock time can be large—pass a shorter `durations` list from Python for smoke tests.

---

When writing up results, cite the literature keys you use from `docs/literature_review.py` plus any extra references you add in your bibliography.
