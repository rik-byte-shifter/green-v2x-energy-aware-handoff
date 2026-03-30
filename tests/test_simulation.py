import math
import os
import sys
from statistics import NormalDist

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulations.config import SimulationConfig
from simulations.simulator import V2XSimulator
from src.models.vehicle import Vehicle
from src.models.basestation import BaseStation, BSConfig
from src.models.energy import EnvironmentalMetrics
from src.algorithms.energy_aware_handoff import EnergyAwareHandoff


def test_environmental_metrics_co2():
    em = EnvironmentalMetrics(carbon_intensity_kg_per_kwh=0.5)
    assert em.energy_to_co2(3.6e6) == pytest.approx(0.5)
    v = Vehicle(0, 0.0, 0.0, speed=10.0, direction=0.0)
    v.state.energy_consumed = 3.6e6
    fp = em.calculate_vehicle_co2_footprint(v)
    assert fp["energy_joules"] == pytest.approx(3.6e6)
    assert fp["co2_kg"] == pytest.approx(0.5)
    assert fp["co2_grams"] == pytest.approx(500.0)

    # 1 kg fleet CO2, 10 vehicles, 100 s sim, 1000 s/year -> 0.1 kg/veh/sim * 10 = 1 kg/veh/yr
    y = em.co2_kg_per_vehicle_per_year(1.0, 10, 100.0, 1000.0)
    assert y == pytest.approx(1.0)
    assert em.avg_co2_kg_per_vehicle(1.0, 10) == pytest.approx(0.1)


def test_weather_profile_resolution():
    cfg = SimulationConfig(weather_profile="heavy_rain")
    wp = cfg.get_weather()
    assert wp.name == "Heavy Rain"
    assert wp.path_loss_exponent == pytest.approx(4.0)
    assert wp.rain_attenuation_db_per_km == pytest.approx(7.0)


def test_rain_attenuation_is_added_to_path_loss():
    # With identical configs except rain attenuation, the difference in path-loss
    # at 1 km should equal the attenuation term (dB/km * km).
    cfg_no = BSConfig(
        coverage_radius=100.0,
        shadowing_std_db=0.0,
        path_loss_exponent=2.0,
        rain_attenuation_db_per_km=0.0,
    )
    cfg_yes = BSConfig(
        coverage_radius=100.0,
        shadowing_std_db=0.0,
        path_loss_exponent=2.0,
        rain_attenuation_db_per_km=10.0,
    )

    bs_no = BaseStation(0, x=0.0, y=0.0, config=cfg_no)
    bs_yes = BaseStation(1, x=0.0, y=0.0, config=cfg_yes)

    d_m = 1000.0  # 1 km
    pl_no = bs_no.calculate_path_loss(d_m)
    pl_yes = bs_yes.calculate_path_loss(d_m)
    assert pl_yes - pl_no == pytest.approx(10.0)


def test_percentile_margin_increases_required_tx_power():
    # Margin model: Ptx = Prx_target + PL + z_p*sigma. Verify delta in dB.
    sigma_db = 10.0
    p = 0.95
    z = NormalDist().inv_cdf(p)
    expected_margin_db = z * sigma_db

    cfg_base = BSConfig(
        coverage_radius=1000.0,
        shadowing_std_db=sigma_db,
        path_loss_exponent=2.0,
        rain_attenuation_db_per_km=0.0,
        shadowing_reliability=0.5,  # z=0 -> zero margin baseline
        tx_power_min_watts=1e-9,
        tx_power_max_watts=10.0,
    )
    cfg_margin = BSConfig(
        coverage_radius=1000.0,
        shadowing_std_db=sigma_db,
        path_loss_exponent=2.0,
        rain_attenuation_db_per_km=0.0,
        shadowing_reliability=p,
        tx_power_min_watts=1e-9,
        tx_power_max_watts=10.0,
    )
    bs0 = BaseStation(0, x=0.0, y=0.0, config=cfg_base)
    bs1 = BaseStation(1, x=0.0, y=0.0, config=cfg_margin)

    d_m = 100.0
    p0 = bs0.calculate_tx_power_required_for_target_rx(d_m)
    p1 = bs1.calculate_tx_power_required_for_target_rx(d_m)
    delta_db = 10.0 * math.log10(p1 / p0)
    assert delta_db == pytest.approx(expected_margin_db, rel=1e-6)


def test_energy_aware_qos_filter_and_fallback():
    _cfg = BSConfig(coverage_radius=800.0)
    v = Vehicle(0, x=500, y=500, speed=20.0)
    bs1 = BaseStation(0, x=400, y=400, config=_cfg)
    bs2 = BaseStation(1, x=1200, y=400, config=_cfg)
    ea = EnergyAwareHandoff(min_data_rate_bps=5e6)
    bs, info = ea.select_best_bs(v, [bs1, bs2])
    assert bs is not None
    assert info.get("qos_met") is True
    assert info["min_data_rate_bps"] == 5e6
    assert info["data_rate"] >= 5e6

    ea_fallback = EnergyAwareHandoff(min_data_rate_bps=1e20)
    _, info_fb = ea_fallback.select_best_bs(v, [bs1, bs2])
    assert info_fb.get("qos_met") is False


def test_vehicle_distance():
    v = Vehicle(0, 0.0, 0.0, speed=10.0, direction=0.0)
    assert v.distance_to(3.0, 4.0) == pytest.approx(5.0)


def test_highway_lane_speed_defaults_increase_with_lane_index():
    cfg = SimulationConfig(
        movement_mode="highway",
        highway_num_lanes=4,
        area_size=500,
    )
    lo0, hi0 = cfg.highway_lane_speed_bounds(0)
    lo3, hi3 = cfg.highway_lane_speed_bounds(3)
    assert lo3 >= lo0 and hi3 >= hi0


def test_highway_lane_straight_and_wrap():
    lane = 100.0
    v = Vehicle(
        0, 0.0, lane, speed=10.0, lane_y=lane, highway_direction_rad=0.0
    )
    v.move(1.0, boundary=(0, 1000))
    assert v.y == pytest.approx(lane)
    assert v.x == pytest.approx(10.0)
    v.x = 995.0
    v.move(1.0, boundary=(0, 1000))
    assert v.y == pytest.approx(lane)
    assert v.x == pytest.approx(5.0)


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
    assert 'co2_kg' in out['stats']
    assert out['stats']['co2_kg'] >= 0.0
    assert 'co2_kg_per_vehicle_per_year' in out['stats']
    assert out['stats']['co2_kg_per_vehicle_per_year'] >= 0.0
    assert 'avg_throughput_bps' in out['stats']
    assert out['stats']['avg_throughput_bps'] >= 0.0
    assert 'p5_throughput_bps' in out['stats']
    assert out['stats']['p5_throughput_bps'] >= 0.0
    assert 'outage_probability_percent' in out['stats']
    assert 0.0 <= out['stats']['outage_probability_percent'] <= 100.0
    assert 'ping_pong_handoffs' in out['stats']
    assert out['stats']['ping_pong_handoffs'] >= 0
    assert 'ping_pong_rate_percent' in out['stats']
    assert 0.0 <= out['stats']['ping_pong_rate_percent'] <= 100.0


def test_stronger_baselines_short_run():
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
    for name in ("sinr", "load_aware_rssi"):
        out = sim.run_algorithm(name)
        assert "stats" in out
        assert out["stats"]["total_bits"] >= 0
        assert out["stats"]["avg_energy_per_bit"] >= 0.0
        assert out["stats"]["avg_throughput_bps"] >= 0.0
        assert 0.0 <= out["stats"]["outage_probability_percent"] <= 100.0


def test_link_metrics_include_interference_and_sinr():
    cfg = SimulationConfig(
        num_vehicles=1,
        num_base_stations=4,
        duration=1,
        area_size=500,
        seed=7,
    )
    sim = V2XSimulator(cfg)
    sim.setup_network()
    sim.setup_vehicles()
    v = sim.vehicles[0]
    get_link = sim._make_step_link_metrics_getter()
    rows = [get_link(v, bs, False) for bs in sim.base_stations]
    rows = [r for r in rows if r is not None]
    assert rows, "At least one BS should produce link metrics"
    assert all("interference_mw" in r for r in rows)
    assert all(r["interference_mw"] >= 0.0 for r in rows)
