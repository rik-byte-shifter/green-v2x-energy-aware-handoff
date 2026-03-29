"""Aggregate metrics helpers for simulation outputs."""


def format_energy_per_bit_j_per_bit(epb: float) -> str:
    if epb <= 0 or epb == float('inf'):
        return "N/A"
    return f"{epb * 1e6:.4f} uJ/bit"
