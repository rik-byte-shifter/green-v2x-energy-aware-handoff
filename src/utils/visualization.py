import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.models.basestation import BaseStation
from src.models.vehicle import Vehicle


class ResultVisualizer:
    """Generate publication-quality plots"""

    def __init__(self, results: Dict, results_dir: str = 'results'):
        self.results = results
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12

    def plot_energy_comparison(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ea_metrics = self.results['energy_aware']['metrics']
        rssi_metrics = self.results['rssi']['metrics']
        naive_metrics = self.results.get('naive_nearest', {}).get('metrics')
        time = ea_metrics['time']

        axes[0, 0].plot(time, ea_metrics['avg_energy_per_bit'],
                       'g-', linewidth=2, label='Energy-Aware')
        axes[0, 0].plot(time, rssi_metrics['avg_energy_per_bit'],
                       'r--', linewidth=2, label='RSSI-Based')
        if naive_metrics:
            axes[0, 0].plot(time, naive_metrics['avg_energy_per_bit'],
                           color='#ff7f0e', linestyle=':', linewidth=2, label='Naive Nearest')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Energy-per-Bit (J/bit)')
        axes[0, 0].set_title('Energy Efficiency Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(time, ea_metrics['avg_tx_power'],
                       'g-', linewidth=2, label='Energy-Aware')
        axes[0, 1].plot(time, rssi_metrics['avg_tx_power'],
                       'r--', linewidth=2, label='RSSI-Based')
        if naive_metrics:
            axes[0, 1].plot(time, naive_metrics['avg_tx_power'],
                           color='#ff7f0e', linestyle=':', linewidth=2, label='Naive Nearest')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Avg TX Power (W)')
        axes[0, 1].set_title('Transmission Power Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(time, ea_metrics['handoffs'],
                       'g-', linewidth=2, label='Energy-Aware')
        axes[1, 0].plot(time, rssi_metrics['handoffs'],
                       'r--', linewidth=2, label='RSSI-Based')
        if naive_metrics:
            axes[1, 0].plot(time, naive_metrics['handoffs'],
                           color='#ff7f0e', linestyle=':', linewidth=2, label='Naive Nearest')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Total Handoffs')
        axes[1, 0].set_title('Cumulative Handoffs')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(time, ea_metrics['connected_vehicles'],
                       'g-', linewidth=2, label='Energy-Aware')
        axes[1, 1].plot(time, rssi_metrics['connected_vehicles'],
                       'r--', linewidth=2, label='RSSI-Based')
        if naive_metrics:
            axes[1, 1].plot(time, naive_metrics['connected_vehicles'],
                           color='#ff7f0e', linestyle=':', linewidth=2, label='Naive Nearest')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Connected Vehicles')
        axes[1, 1].set_title('Network Connectivity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(self.results_dir, 'energy_comparison.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _carbon_intensity(self) -> float:
        s = self.results['energy_aware']['stats']
        return float(s.get('carbon_intensity_kg_per_kwh', 0.5))

    def plot_cumulative_co2(self):
        """Cumulative operational CO2 (kg) from step energies × grid intensity."""
        ea_metrics = self.results['energy_aware']['metrics']
        rssi_metrics = self.results['rssi']['metrics']
        naive_metrics = self.results.get('naive_nearest', {}).get('metrics')
        time = ea_metrics['time']
        ci = self._carbon_intensity()
        j_to_kg = ci / 3.6e6

        def cum_co2(step_joules_series):
            c = np.cumsum(np.asarray(step_joules_series, dtype=float) * j_to_kg)
            return c

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time, cum_co2(ea_metrics['total_energy']), 'g-', lw=2, label='Energy-Aware')
        ax.plot(time, cum_co2(rssi_metrics['total_energy']), 'r--', lw=2, label='RSSI-Based')
        if naive_metrics:
            ax.plot(
                time,
                cum_co2(naive_metrics['total_energy']),
                color='#ff7f0e',
                ls=':',
                lw=2,
                label='Naive Nearest',
            )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cumulative CO2 (kg)')
        ax.set_title(f'Cumulative Carbon Emissions (intensity {ci:g} kg CO2/kWh)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(self.results_dir, 'cumulative_co2.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_per_vehicle_energy_cdf(self):
        """Empirical CDF of per-vehicle total energy (Joules)."""
        ea_stats = self.results['energy_aware']['stats']
        rssi_stats = self.results['rssi']['stats']
        naive_stats = self.results.get('naive_nearest', {}).get('stats')
        ea_e = np.sort(np.asarray(ea_stats['per_vehicle_energy_joules']))
        r_e = np.sort(np.asarray(rssi_stats['per_vehicle_energy_joules']))
        n = len(ea_e)
        y = (np.arange(1, n + 1, dtype=float)) / n

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.step(ea_e, y, where='post', color='green', lw=2, label='Energy-Aware')
        ax.step(r_e, y, where='post', color='red', ls='--', lw=2, label='RSSI-Based')
        if naive_stats:
            n_e = np.sort(np.asarray(naive_stats['per_vehicle_energy_joules']))
            ax.step(n_e, y, where='post', color='#ff7f0e', ls=':', lw=2, label='Naive Nearest')
        ax.set_xlabel('Per-vehicle total energy (J)')
        ax.set_ylabel('CDF')
        ax.set_title('CDF of Energy Consumption (per vehicle)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(self.results_dir, 'energy_per_vehicle_cdf.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_bar_comparison(self):
        ea_stats = self.results['energy_aware']['stats']
        rssi_stats = self.results['rssi']['stats']
        naive_stats = self.results.get('naive_nearest', {}).get('stats')

        labels = ['Energy/Bit (uJ)', 'TX Power (mW)', 'Handoffs']

        ea_values = [
            ea_stats['avg_energy_per_bit'] * 1e6,
            ea_stats['avg_tx_power'] * 1000,
            ea_stats['total_handoffs']
        ]

        rssi_values = [
            rssi_stats['avg_energy_per_bit'] * 1e6,
            rssi_stats['avg_tx_power'] * 1000,
            rssi_stats['total_handoffs']
        ]

        naive_values = None
        if naive_stats:
            naive_values = [
                naive_stats['avg_energy_per_bit'] * 1e6,
                naive_stats['avg_tx_power'] * 1000,
                naive_stats['total_handoffs']
            ]

        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width, ea_values, width, label='Energy-Aware', color='green', alpha=0.8)
        bars2 = ax.bar(x, rssi_values, width, label='RSSI-Based', color='red', alpha=0.8)
        bars3 = None
        if naive_values:
            bars3 = ax.bar(x + width, naive_values, width, label='Naive Nearest', color='#ff7f0e', alpha=0.85)

        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title('Algorithm Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

        if bars3:
            for bar in bars3:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        out = os.path.join(self.results_dir, 'bar_comparison.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_bar_co2_comparison(self):
        ea_stats = self.results['energy_aware']['stats']
        rssi_stats = self.results['rssi']['stats']
        naive_stats = self.results.get('naive_nearest', {}).get('stats')

        labels = ['Total CO2 (g)', 'CO2 / veh / yr (g)']
        ea_vals = [ea_stats['co2_grams'], ea_stats['co2_kg_per_vehicle_per_year'] * 1000.0]
        r_vals = [rssi_stats['co2_grams'], rssi_stats['co2_kg_per_vehicle_per_year'] * 1000.0]
        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, ea_vals, width, label='Energy-Aware', color='green', alpha=0.8)
        ax.bar(x, r_vals, width, label='RSSI-Based', color='red', alpha=0.8)
        if naive_stats:
            n_vals = [
                naive_stats['co2_grams'],
                naive_stats['co2_kg_per_vehicle_per_year'] * 1000.0,
            ]
            ax.bar(x + width, n_vals, width, label='Naive Nearest', color='#ff7f0e', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Grams CO2 (reporting)')
        ax.set_title('Carbon Emissions (operational, grid intensity from config)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        out = os.path.join(self.results_dir, 'bar_co2_comparison.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_network_topology(self, vehicles: List[Vehicle], base_stations: List[BaseStation]):
        fig, ax = plt.subplots(figsize=(10, 10))

        bs_x = [bs.x for bs in base_stations]
        bs_y = [bs.y for bs in base_stations]
        ax.scatter(bs_x, bs_y, c='blue', s=200, marker='^',
                  label='Base Stations', zorder=5, edgecolors='darkblue')

        for bs in base_stations:
            circle = plt.Circle((bs.x, bs.y), bs.config.coverage_radius,
                              color='blue', fill=False, alpha=0.3, linewidth=1)
            ax.add_patch(circle)

        v_x = [v.x for v in vehicles]
        v_y = [v.y for v in vehicles]
        ax.scatter(v_x, v_y, c='green', s=100, marker='o',
                  label='Vehicles', zorder=5, alpha=0.7)

        for v in vehicles:
            if v.state.connected_bs_id is not None:
                bs = next(bs for bs in base_stations if bs.bs_id == v.state.connected_bs_id)
                ax.plot([v.x, bs.x], [v.y, bs.y], 'g-', alpha=0.3, linewidth=1)

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('V2X Network Topology')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        out = os.path.join(self.results_dir, 'network_topology.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_tx_power_distribution(self):
        ea_stats = self.results['energy_aware']['stats']
        rssi_stats = self.results['rssi']['stats']
        ea = np.asarray(ea_stats.get('tx_power_samples_w', []), dtype=float) * 1000.0
        rssi = np.asarray(rssi_stats.get('tx_power_samples_w', []), dtype=float) * 1000.0
        if ea.size == 0 or rssi.size == 0:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ea, bins=40, density=True, alpha=0.5, color='green', label='Energy-Aware')
        ax.hist(rssi, bins=40, density=True, alpha=0.5, color='red', label='RSSI-Based')
        ax.set_xlabel('TX power (mW)')
        ax.set_ylabel('Density')
        ax.set_title('TX Power Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(self.results_dir, 'tx_power_distribution.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_bs_load_distribution(self):
        ea_stats = self.results['energy_aware']['stats']
        rssi_stats = self.results['rssi']['stats']
        ea = np.asarray(ea_stats.get('bs_load_samples', []), dtype=float)
        rssi = np.asarray(rssi_stats.get('bs_load_samples', []), dtype=float)
        if ea.size == 0 or rssi.size == 0:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ea, bins=30, density=True, alpha=0.5, color='green', label='Energy-Aware')
        ax.hist(rssi, bins=30, density=True, alpha=0.5, color='red', label='RSSI-Based')
        ax.set_xlabel('BS load (0..1)')
        ax.set_ylabel('Density')
        ax.set_title('Load per Base Station Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(self.results_dir, 'bs_load_distribution.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_sinr_histogram(self):
        ea_stats = self.results['energy_aware']['stats']
        rssi_stats = self.results['rssi']['stats']
        ea = np.asarray(ea_stats.get('sinr_samples_db', []), dtype=float)
        rssi = np.asarray(rssi_stats.get('sinr_samples_db', []), dtype=float)
        if ea.size == 0 or rssi.size == 0:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ea, bins=40, density=True, alpha=0.5, color='green', label='Energy-Aware')
        ax.hist(rssi, bins=40, density=True, alpha=0.5, color='red', label='RSSI-Based')
        ax.set_xlabel('SINR (dB)')
        ax.set_ylabel('Density')
        ax.set_title('SINR Histogram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(self.results_dir, 'sinr_histogram.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def generate_all_plots(self, vehicles: Optional[List[Vehicle]] = None,
                          base_stations: Optional[List[BaseStation]] = None):
        print("\nGenerating visualization plots...")

        self.plot_energy_comparison()
        self.plot_cumulative_co2()
        self.plot_bar_comparison()
        self.plot_bar_co2_comparison()
        self.plot_per_vehicle_energy_cdf()
        self.plot_tx_power_distribution()
        self.plot_bs_load_distribution()
        self.plot_sinr_histogram()

        if vehicles is not None and base_stations is not None:
            self.plot_network_topology(vehicles, base_stations)

        print(f"All plots saved to {self.results_dir}/")
