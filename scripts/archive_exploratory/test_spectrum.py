import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scripts.hawking_detection_experiment import calculate_hawking_spectrum

from analog_hawking.detection.radio_snr import (
    band_power_from_spectrum,
    equivalent_signal_temperature,
)

kappa = 1e12
result = calculate_hawking_spectrum(kappa, n_points=1000, emitting_area_m2=1e-6, solid_angle_sr=0.05, coupling_efficiency=0.1)

print(f"Success: {result.get('success')}")
print(f"Temperature: {result.get('temperature')} K")
print(f"Peak frequency: {result.get('peak_frequency')} Hz")
print(f"Frequencies range: {result['frequencies'][0]:.3e} to {result['frequencies'][-1]:.3e} Hz")

P_sig = band_power_from_spectrum(result['frequencies'], result['power_spectrum'], result['peak_frequency'], 1e8)
T_sig = equivalent_signal_temperature(P_sig, 1e8)
print(f"P_sig: {P_sig:.3e} W")
print(f"T_sig: {T_sig:.3e} K")
