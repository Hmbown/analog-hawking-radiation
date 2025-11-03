import numpy as np
from scipy.constants import k

from analog_hawking.detection.radio_snr import (
    band_power_from_spectrum,
    equivalent_signal_temperature,
)


def test_band_power_uniform_psd_matches_area():
    # Construct a frequency grid around f0 with uniform PSD A inside the band
    f0 = 2.0e8  # 200 MHz
    B = 1.0e8  # 100 MHz
    A = 1.0e-22  # W/Hz (uniform PSD)

    f = np.linspace(f0 - 2 * B, f0 + 2 * B, 10_000)
    psd = np.zeros_like(f)
    mask = (f >= f0 - B / 2) & (f <= f0 + B / 2)
    psd[mask] = A

    P = band_power_from_spectrum(f, psd, f0, B)
    # Exact integral of constant A over bandwidth B is A*B (discrete approx tolerates <1%)
    assert np.isclose(P, A * B, rtol=1e-2)


def test_equivalent_temperature_from_uniform_psd():
    B = 1.0e8
    A = 1.0e-22
    # In-band power for uniform PSD is A*B
    P = A * B
    T_sig = equivalent_signal_temperature(P, B)
    # T_sig = P/(kB*B) = A/kB
    assert np.isclose(T_sig, A / k, rtol=1e-12)
