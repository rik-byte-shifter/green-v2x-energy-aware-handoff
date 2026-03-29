import numpy as np


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
            fading = np.random.rayleigh()
        elif self.fading_type == 'rician':
            k_factor = 10
            fading = np.sqrt(np.random.normal(np.sqrt(k_factor), 1) ** 2 +
                           np.random.normal(0, 1) ** 2)
        else:
            fading = 1.0

        return signal_power * (fading ** 2)

    def calculate_snr(self, rx_power_dbm: float,
                     noise_power_dbm: float = -174.0) -> float:
        bandwidth_hz = 20e6
        total_noise = noise_power_dbm + 10 * np.log10(bandwidth_hz)

        snr = rx_power_dbm - total_noise
        return snr

    def calculate_data_rate(self, snr_db: float,
                           bandwidth: float = 20e6) -> float:
        snr_linear = 10 ** (snr_db / 10.0)
        data_rate = bandwidth * np.log2(1 + snr_linear)

        return data_rate
