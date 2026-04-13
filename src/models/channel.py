import numpy as np

# 20 MHz NR/LTE-style discrete MCS ceilings (bps): SNR must reach threshold to use rate
MCS_MIN_SNR_DB = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0], dtype=float)
MCS_MAX_RATE_BPS = np.array(
    [3.0e6, 8.0e6, 16.0e6, 32.0e6, 55.0e6, 80.0e6, 100.0e6], dtype=float
)
DEFAULT_BANDWIDTH_HZ = 20e6
DEFAULT_MAX_DATA_RATE_BPS = 100e6


def bounded_shannon_data_rate(
    snr_db: float,
    bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ,
    max_rate_bps: float = DEFAULT_MAX_DATA_RATE_BPS,
    min_snr_db: float = 0.0,
) -> float:
    """
    Shannon capacity limited by discrete MCS and a physical max rate.

    Uses C = B log2(1 + SNR_linear), then min with the highest MCS rate the
    SNR supports, then caps at max_rate_bps.

    If SNR is at or below min_snr_db (dB), the link is in outage (rate 0).
    """
    if snr_db <= min_snr_db:
        return 0.0
    snr_linear = 10 ** (snr_db / 10.0)
    shannon = bandwidth_hz * np.log2(1 + snr_linear)
    idx = int(np.searchsorted(MCS_MIN_SNR_DB, snr_db, side="right")) - 1
    if idx < 0:
        return 0.0
    mcs_ceiling = float(MCS_MAX_RATE_BPS[idx])
    return float(min(shannon, mcs_ceiling, max_rate_bps))


def apply_log_normal_shadowing(rx_power_dbm: float, shadowing_std_db: float) -> float:
    """
    Add slow (log-normal) shadowing in the dB domain to received power.
    shadowing_std_db: standard deviation in dB (typ. 6–10). 0 = disabled.
    """
    if shadowing_std_db <= 0:
        return rx_power_dbm
    return float(rx_power_dbm + np.random.normal(0.0, shadowing_std_db))


class ChannelModel:
    """
    Wireless Channel Model for V2X

    Includes:
    - Path loss
    - Shadowing (log-normal)
    - Fast fading (Rayleigh/Rician)
    """

    def __init__(self, path_loss_exponent: float = 3.0,
                 shadowing_std: float = 8.0,
                 fading_type: str = 'rayleigh'):
        self.path_loss_exponent = path_loss_exponent
        self.shadowing_std = shadowing_std
        self.fading_type = fading_type

    def calculate_path_loss(self, distance: float,
                           frequency: float = 2.4e9) -> float:
        d0 = 1.0
        pl_d0 = 20 * np.log10(4 * np.pi * d0 * frequency / 3e8)

        path_loss = pl_d0 + 10 * self.path_loss_exponent * \
                    np.log10(distance / d0)

        return path_loss

    def add_shadowing(self, path_loss: float) -> float:
        """Add log-normal shadowing"""
        shadowing = np.random.normal(0, self.shadowing_std)
        return path_loss + shadowing

    def add_fading(self, signal_power: float) -> float:
        if self.fading_type == 'rayleigh':
            # Normalized Rayleigh amplitude: E[|h|^2] = 1
            fading = np.random.rayleigh(scale=1.0 / np.sqrt(2.0))
        elif self.fading_type == 'rician':
            # Normalized Rician amplitude with K-factor (linear).
            k_factor = 10.0
            los_mean = np.sqrt(k_factor / (k_factor + 1.0))
            scatter_std = np.sqrt(1.0 / (2.0 * (k_factor + 1.0)))
            i = np.random.normal(los_mean, scatter_std)
            q = np.random.normal(0.0, scatter_std)
            fading = np.sqrt(i ** 2 + q ** 2)
        else:
            fading = 1.0

        return signal_power * (fading ** 2)

    def calculate_snr(self, rx_power_dbm: float,
                     noise_power_dbm: float = -174.0) -> float:
        bandwidth_hz = 20e6
        total_noise = noise_power_dbm + 10 * np.log10(bandwidth_hz)

        snr = rx_power_dbm - total_noise
        return snr

    def calculate_data_rate(
        self,
        snr_db: float,
        bandwidth: float = DEFAULT_BANDWIDTH_HZ,
        max_rate_bps: float = DEFAULT_MAX_DATA_RATE_BPS,
        min_snr_db: float = 0.0,
    ) -> float:
        return bounded_shannon_data_rate(snr_db, bandwidth, max_rate_bps, min_snr_db)
