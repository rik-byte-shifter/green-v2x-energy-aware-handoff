"""
Multi-weather sweep with uncertainty across seeds.

Produces:
  - results/weather_sweep.json
  - results/weather_sweep_energy_saving.png
  - results/weather_sweep_handoff_reduction.png
  - results/weather_sweep_co2_saving.png
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from typing import Dict, Any, List

import numpy as np

from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulations.config import SimulationConfig
from simulations.simulator import V2XSimulator
from src.models.weather import WEATHER_PROFILES


SEEDS_DEFAULT = [42, 123, 456, 789, 1011]


def _mean_and_ci95(values: List[float]) -> Dict[str, float]:
    """Compute mean and 95% CI using t-distribution over seed samples."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    mean = float(np.mean(arr))
    if n == 1:
        return {
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
        }
    std = float(np.std(arr, ddof=1))
    se = std / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    half = tcrit * se
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _as_label(profile_key: str) -> str:
    p = WEATHER_PROFILES[profile_key]
    # Keep labels short for bar plots.
    return p.name


def _extract_comparison_metrics(comparison: Dict[str, Any]) -> Dict[str, float]:
    # Required metrics for the requested evidence.
    return {
        "energy_saving_percent": float(comparison.get("energy_saving_percent", 0.0)),
        "handoff_reduction_percent": float(
            comparison.get("handoff_reduction_percent", 0.0)
        ),
        "co2_saving_percent": float(comparison.get("co2_saving_percent", 0.0)),
        # Optional (but useful) PHY-aware averages for context.
        "avg_energy_per_bit_ea": float(
            comparison.get("energy_aware_stats", {}).get("avg_energy_per_bit", 0.0)
        ),
        "avg_data_rate_ea": float(
            comparison.get("energy_aware_stats", {}).get("avg_data_rate", 0.0)
        ),
    }


def _plot_metric_bars(
    *,
    out_path: str,
    per_weather_summary: Dict[str, Dict[str, float]],
    metric_key: str,
    title: str,
    ylabel: str,
):
    weather_keys = list(per_weather_summary.keys())
    labels = [_as_label(wk) for wk in weather_keys]

    means = [per_weather_summary[wk][metric_key]["mean"] for wk in weather_keys]
    lows = [per_weather_summary[wk][metric_key]["ci95_low"] for wk in weather_keys]
    highs = [per_weather_summary[wk][metric_key]["ci95_high"] for wk in weather_keys]

    means_arr = np.asarray(means, dtype=float)
    err_low = means_arr - np.asarray(lows, dtype=float)
    err_high = np.asarray(highs, dtype=float) - means_arr

    # For asymmetric error bars.
    yerr = np.vstack([err_low, err_high])

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(weather_keys))
    ax.bar(x, means_arr, yerr=yerr, capsize=4, alpha=0.85, color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_weather_sweep(
    *,
    seeds: List[int],
    weather_profiles: List[str],
    duration: int,
    time_step: float,
    num_vehicles: int,
    num_base_stations: int,
    area_size: int,
    results_dir: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    os.makedirs(results_dir, exist_ok=True)

    per_weather: Dict[str, Any] = {}

    for wk in weather_profiles:
        p = WEATHER_PROFILES[wk]
        per_weather[wk] = {
            "profile_name": p.name,
            "per_seed": {},
        }

        for seed in seeds:
            cfg = SimulationConfig.paper_baseline_scenario()
            cfg.weather_profile = wk  # weather key
            cfg.seed = seed
            # Allow overrides for faster experimentation.
            cfg.duration = duration
            cfg.time_step = float(time_step)
            cfg.num_vehicles = int(num_vehicles)
            cfg.num_base_stations = int(num_base_stations)
            cfg.area_size = int(area_size)

            sim = V2XSimulator(cfg)
            if verbose:
                comparison = sim.run_comparison()
            else:
                # The simulator prints a lot per run; suppress by default.
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
                        devnull
                    ):
                        comparison = sim.run_comparison()

            per_weather[wk]["per_seed"][str(seed)] = _extract_comparison_metrics(
                comparison
            )

    # Compute summary (mean and 95% CI across seeds) for each metric.
    metric_keys = [
        "energy_saving_percent",
        "handoff_reduction_percent",
        "co2_saving_percent",
        "avg_energy_per_bit_ea",
        "avg_data_rate_ea",
    ]

    for wk in weather_profiles:
        seed_metrics = per_weather[wk]["per_seed"]
        for mk in metric_keys:
            vals = [
                seed_metrics[str(seed)][mk]
                for seed in seeds
                if str(seed) in seed_metrics and mk in seed_metrics[str(seed)]
            ]
            if mk not in per_weather[wk]:
                per_weather[wk][mk] = {}
            per_weather[wk][mk] = _mean_and_ci95(vals)

    return per_weather


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(map(str, SEEDS_DEFAULT)),
        help="Comma-separated seed list (default: journal seeds).",
    )
    parser.add_argument(
        "--weathers",
        type=str,
        default=",".join(WEATHER_PROFILES.keys()),
        help="Comma-separated weather profile keys to run.",
    )
    parser.add_argument("--duration", type=int, default=800)
    parser.add_argument("--time-step", type=float, default=1.0)
    parser.add_argument("--num-vehicles", type=int, default=20)
    parser.add_argument("--num-base-stations", type=int, default=9)
    parser.add_argument("--area-size", type=int, default=3000)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print simulator logs for each weather/seed run (very verbose).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for JSON/plots.",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    weathers = [w.strip() for w in args.weathers.split(",") if w.strip()]

    # Validate weather keys early.
    invalid = [w for w in weathers if w not in WEATHER_PROFILES]
    if invalid:
        raise ValueError(f"Unknown weather profile(s): {invalid}.")

    # Keep the requested order for plots.
    weather_profiles = [w for w in WEATHER_PROFILES.keys() if w in weathers]

    sweep = run_weather_sweep(
        seeds=seeds,
        weather_profiles=weather_profiles,
        duration=args.duration,
        time_step=args.time_step,
        num_vehicles=args.num_vehicles,
        num_base_stations=args.num_base_stations,
        area_size=args.area_size,
        results_dir=args.results_dir,
        verbose=bool(args.verbose),
    )

    out_json = os.path.join(args.results_dir, "weather_sweep.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)

    # Build plotting structure expected by _plot_metric_bars.
    per_weather_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for wk in weather_profiles:
        per_weather_summary[wk] = {
            "energy_saving_percent": sweep[wk]["energy_saving_percent"],
            "handoff_reduction_percent": sweep[wk]["handoff_reduction_percent"],
            "co2_saving_percent": sweep[wk]["co2_saving_percent"],
            # Not required by request, but we compute anyway.
            "avg_energy_per_bit_ea": sweep[wk]["avg_energy_per_bit_ea"],
            "avg_data_rate_ea": sweep[wk]["avg_data_rate_ea"],
        }

    _plot_metric_bars(
        out_path=os.path.join(args.results_dir, "weather_sweep_energy_saving.png"),
        per_weather_summary=per_weather_summary,
        metric_key="energy_saving_percent",
        title="EA vs RSSI: Energy Saving vs Weather (95% CI across seeds)",
        ylabel="Energy saving (%)",
    )
    _plot_metric_bars(
        out_path=os.path.join(
            args.results_dir, "weather_sweep_handoff_reduction.png"
        ),
        per_weather_summary=per_weather_summary,
        metric_key="handoff_reduction_percent",
        title="EA vs RSSI: Handoff Reduction vs Weather (95% CI across seeds)",
        ylabel="Handoff reduction (%)",
    )
    _plot_metric_bars(
        out_path=os.path.join(args.results_dir, "weather_sweep_co2_saving.png"),
        per_weather_summary=per_weather_summary,
        metric_key="co2_saving_percent",
        title="EA vs RSSI: CO2 Saving vs Weather (95% CI across seeds)",
        ylabel="CO2 saving (%)",
    )

    print(f"Weather sweep complete. JSON: {out_json}")
    print(
        "Plots: weather_sweep_energy_saving.png, weather_sweep_handoff_reduction.png, weather_sweep_co2_saving.png"
    )


if __name__ == "__main__":
    main()

