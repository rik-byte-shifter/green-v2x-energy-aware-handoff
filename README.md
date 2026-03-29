# Green V2X: Energy-Aware Handoff

Python simulation comparing **RSSI-based** handoff with an **energy-aware** handoff that minimizes an energy-per-bit style metric (including load penalty) with hysteresis.

## Setup

```powershell
cd green_v2x_project
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

Outputs plots and `results/simulation_results.json`. The working directory is set to the project root automatically.

## Tests

```powershell
python -m pytest tests\test_simulation.py -v
```

## Layout

- `src/models/` — vehicle, base station, channel, energy
- `src/algorithms/` — RSSI and energy-aware handoff
- `simulations/` — config and `V2XSimulator`
- `src/utils/` — metrics and plotting (matplotlib, non-interactive `Agg` backend)
- `notebooks/analysis.ipynb` — optional JSON inspection

## Note

Console logs use `uJ/bit` (microjoules per bit) for compatibility with Windows code pages. Figures may use microjoule labeling in axis text.

### Metrics and tuning

- **Total energy** accumulates **PA + circuit + baseband + cooling** via `EnergyModel.calculate_total_power` during each connected time step, plus a configurable **per-handoff energy** (`SimulationConfig.handoff_energy_joules`) and a **handoff cooldown** (`handoff_cooldown_s`) to limit ping-pong.
- **RSSI baseline** compares strongest received signal using **per-link adaptive TX** (required power to each BS), so it is not the same decision rule as energy-aware **EPB × load** minimization.
- If average TX sits at the **1 mW floor**, link energy looks similar across policies; increase path-loss stress (e.g. larger area, smaller `bs_coverage_radius`, or tune `EnergyParams` / link budget in code) to widen separation for plots and the paper.

