import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulations.config import SimulationConfig
from simulations.simulator import V2XSimulator
from src.models.vehicle import Vehicle
from src.models.basestation import BaseStation, BSConfig


def test_vehicle_distance():
    v = Vehicle(0, 0.0, 0.0, speed=10.0, direction=0.0)
    assert v.distance_to(3.0, 4.0) == pytest.approx(5.0)


def test_bs_coverage():
    bs = BaseStation(0, 0.0, 0.0, BSConfig(coverage_radius=100.0))
    assert bs.is_in_coverage(50.0, 0.0)
    assert not bs.is_in_coverage(200.0, 0.0)


def test_simulator_short_run():
    cfg = SimulationConfig(
        num_vehicles=3,
        num_base_stations=4,
        duration=5,
        area_size=500,
        seed=123,
    )
    sim = V2XSimulator(cfg)
    sim.setup_network()
    sim.setup_vehicles()
    out = sim.run_algorithm('rssi')
    assert 'stats' in out
    assert out['stats']['total_bits'] >= 0
