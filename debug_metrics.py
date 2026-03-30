"""Quick debug script to verify algorithm differentiation"""
from src.models.vehicle import Vehicle
from src.models.basestation import BaseStation, BSConfig
from src.algorithms.energy_aware_handoff import EnergyAwareHandoff
from src.algorithms.rssi_handoff import RSSIHandoff

# Create test scenario (large enough coverage so both BS are candidates)
_dbg_cfg = BSConfig(coverage_radius=800.0)
vehicle = Vehicle(0, x=500, y=500, speed=20)
bs1 = BaseStation(0, x=400, y=400, config=_dbg_cfg)  # Closer
bs2 = BaseStation(1, x=1200, y=400, config=_dbg_cfg)  # Farther

print("=== DEBUG: Algorithm Selection Comparison ===\n")

# Energy-aware selection
ea = EnergyAwareHandoff()
ea_bs, ea_info = ea.select_best_bs(vehicle, [bs1, bs2])
print(f"Energy-Aware selected: BS-{ea_bs.bs_id if ea_bs else None}")
print(f"  TX Power: {ea_info.get('tx_power', 0)*1000:.2f} mW")
print(f"  Energy/Bit: {ea_info.get('energy_per_bit', 0)*1e6:.4f} uJ/bit")
print(f"  Metric: {ea_info.get('metric', 0):.6f}\n")

# RSSI selection
rssi = RSSIHandoff()
rssi_bs, rssi_info = rssi.select_best_bs(vehicle, [bs1, bs2], tx_power=0.1)
print(f"RSSI-Based selected: BS-{rssi_bs.bs_id if rssi_bs else None}")
print(f"  RSSI: {rssi_info.get('rssi', 0):.2f} dBm")
print(f"  Distance: {rssi_info.get('distance', 0):.1f} m\n")

# Check if they differ
if ea_bs != rssi_bs:
    print("[OK] Algorithms selected DIFFERENT base stations!")
else:
    print("[WARN] Both algorithms selected the same BS")
    print("   (This may be correct if one BS is clearly optimal)")
