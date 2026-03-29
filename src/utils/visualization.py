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
        time = ea_metrics['time']

        axes[0, 0].plot(time, ea_metrics['avg_energy_per_bit'],
                       'g-', linewidth=2, label='Energy-Aware')
        axes[0, 0].plot(time, rssi_metrics['avg_energy_per_bit'],
                       'r--', linewidth=2, label='RSSI-Based')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Energy-per-Bit (J/bit)')
        axes[0, 0].set_title('Energy Efficiency Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(time, ea_metrics['avg_tx_power'],
                       'g-', linewidth=2, label='Energy-Aware')
        axes[0, 1].plot(time, rssi_metrics['avg_tx_power'],
                       'r--', linewidth=2, label='RSSI-Based')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Avg TX Power (W)')
        axes[0, 1].set_title('Transmission Power Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(time, ea_metrics['handoffs'],
                       'g-', linewidth=2, label='Energy-Aware')
        axes[1, 0].plot(time, rssi_metrics['handoffs'],
                       'r--', linewidth=2, label='RSSI-Based')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Total Handoffs')
        axes[1, 0].set_title('Cumulative Handoffs')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(time, ea_metrics['connected_vehicles'],
                       'g-', linewidth=2, label='Energy-Aware')
        axes[1, 1].plot(time, rssi_metrics['connected_vehicles'],
                       'r--', linewidth=2, label='RSSI-Based')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Connected Vehicles')
        axes[1, 1].set_title('Network Connectivity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(self.results_dir, 'energy_comparison.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_bar_comparison(self):
        ea_stats = self.results['energy_aware']['stats']
        rssi_stats = self.results['rssi']['stats']

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

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, ea_values, width, label='Energy-Aware', color='green', alpha=0.8)
        bars2 = ax.bar(x + width / 2, rssi_values, width, label='RSSI-Based', color='red', alpha=0.8)

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

        plt.tight_layout()
        out = os.path.join(self.results_dir, 'bar_comparison.png')
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

    def generate_all_plots(self, vehicles: Optional[List[Vehicle]] = None,
                          base_stations: Optional[List[BaseStation]] = None):
        print("\nGenerating visualization plots...")

        self.plot_energy_comparison()
        self.plot_bar_comparison()

        if vehicles is not None and base_stations is not None:
            self.plot_network_topology(vehicles, base_stations)

        print(f"All plots saved to {self.results_dir}/")
