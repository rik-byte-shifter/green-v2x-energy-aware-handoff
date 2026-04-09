"""
verify_fixes.py
---------------
Run this from your project root AFTER applying both fixes:

    python verify_fixes.py

It checks all four problems in ~60 seconds without running the full
800-step multi-seed suite.  Expected output is shown inline.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from dataclasses import replace
from simulations.config import SimulationConfig
from simulations.simulator import V2XSimulator
from src.models.basestation import BSConfig

print("=" * 65)
print("VERIFICATION SCRIPT — checking all 4 fixes")
print("=" * 65)

# ── CHECK 1: overlapping coverage geometry ─────────────────────────
print("\n[1] Coverage geometry check")
cfg = SimulationConfig.sparse_demonstration_scenario()
import math
grid_size = int(math.sqrt(cfg.num_base_stations))
spacing = cfg.area_size // (grid_size + 1)
lane_y = cfg.area_size / 2  # rough lane centre
bs_ys = [(k + 1) * spacing for k in range(grid_size)]
candidates = [y for y in bs_ys if abs(y - lane_y) <= cfg.bs_coverage_radius]
print(f"    num_base_stations : {cfg.num_base_stations}")
print(f"    bs_coverage_radius: {cfg.bs_coverage_radius}m")
print(f"    grid spacing      : {spacing}m")
print(f"    BS y positions    : {bs_ys}")
print(f"    Lane centre y     : {lane_y:.0f}m")
print(f"    BS candidates in range of lane: {len(candidates)}")
if len(candidates) >= 2:
    print("    [PASS] >= 2 candidates → algorithms will make different choices")
else:
    print("    [FAIL] < 2 candidates → baselines will be identical")
    print("           Apply the config.py fix.")

# ── CHECK 2: max_capacity is high enough ───────────────────────────
print("\n[2] BSConfig.max_capacity check")
bs_cfg = BSConfig()
print(f"    BSConfig.max_capacity = {bs_cfg.max_capacity}")
if bs_cfg.max_capacity >= 100:
    print("    [PASS] max_capacity >= 100 → no capacity blocking at 200 vehicles")
else:
    print("    [FAIL] max_capacity too low → capacity block at high vehicle counts")
    print("           Apply the basestation.py fix.")

# ── CHECK 3: run a short comparison and verify baselines differ ────
print("\n[3] Algorithm differentiation check (short run, 30 steps)")
short_cfg = replace(
    SimulationConfig.sparse_demonstration_scenario(),
    duration=30,
    seed=42,
)
sim = V2XSimulator(short_cfg)
sim.run_comparison()

epbs = {
    algo: sim.results[algo]['stats']['avg_energy_per_bit']
    for algo in ['energy_aware', 'rssi', 'sinr', 'load_aware_rssi', 'naive_nearest']
}
print("    avg_energy_per_bit (nJ/bit):")
for algo, epb in epbs.items():
    print(f"      {algo:22s}: {epb * 1e9:.3f}")

rssi_epb   = epbs['rssi']
sinr_epb   = epbs['sinr']
la_epb     = epbs['load_aware_rssi']
ea_epb     = epbs['energy_aware']

rssi_sinr_differ  = abs(rssi_epb - sinr_epb)   > 1e-12
rssi_la_differ    = abs(rssi_epb - la_epb)     > 1e-12
ea_better_rssi    = ea_epb < rssi_epb

print(f"\n    RSSI vs SINR differ    : {rssi_sinr_differ}")
print(f"    RSSI vs Load-aware differ: {rssi_la_differ}")
print(f"    EA better than RSSI    : {ea_better_rssi}")

if rssi_sinr_differ and rssi_la_differ and ea_better_rssi:
    pct = (rssi_epb - ea_epb) / rssi_epb * 100
    print(f"    [PASS] All baselines differ. EA saves {pct:.1f}% EPB vs RSSI.")
else:
    if not rssi_sinr_differ:
        print("    [FAIL] RSSI and SINR still identical.")
    if not rssi_la_differ:
        print("    [FAIL] RSSI and Load-aware RSSI still identical.")
    if not ea_better_rssi:
        print("    [FAIL] Energy-aware not better than RSSI.")

# ── CHECK 4: scaling — no collapse at 200 vehicles ─────────────────
print("\n[4] Scaling check — 200 vehicles (short run)")
scale_cfg = replace(
    SimulationConfig.scaling_scenario(),
    num_vehicles=200,
    duration=30,
    seed=42,
)
sim200 = V2XSimulator(scale_cfg)
sim200.run_comparison()
saving_200 = (
    (sim200.results['rssi']['stats']['avg_energy_per_bit'] -
     sim200.results['energy_aware']['stats']['avg_energy_per_bit'])
    / sim200.results['rssi']['stats']['avg_energy_per_bit'] * 100
)
print(f"    Energy saving at 200 vehicles: {saving_200:.2f}%")
if saving_200 > 1.0:
    print("    [PASS] Non-zero saving at 200 vehicles.")
else:
    print("    [FAIL] Still collapsing to 0% at 200 vehicles.")
    print("           Check that max_capacity=100 is in basestation.py.")

# ── SUMMARY ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Run `python main.py` once all checks show [PASS].")
print("=" * 65)
