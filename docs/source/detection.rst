Detection Module Documentation
==============================

The detection module provides tools for modeling and analyzing the detectability of analog Hawking radiation signatures.

Overview
--------

The detection module focuses on calculating signal-to-noise ratios and integration times for detecting thermal radiation signatures from analog horizons. It implements the radiometer equation and related models for radio astronomy detection.

radio_snr
---------

.. automodule:: analog_hawking.detection.radio_snr
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

band_power_from_spectrum
++++++++++++++++++++++++

Calculate the integrated power within a specified frequency band from a power spectrum.

equivalent_signal_temperature
+++++++++++++++++++++++++++++

Convert in-band power to equivalent antenna temperature using the radiometer relation.

sweep_time_for_5sigma
+++++++++++++++++++++

Compute integration time grids for 5σ detection using the radiometer equation.

Theoretical Background
~~~~~~~~~~~~~~~~~~~~~~

The detection modeling is based on the radiometer equation:

.. math::

   \text{SNR} = \frac{T_{\text{sig}}}{T_{\text{sys}}} \sqrt{B \cdot t}

Where:
- :math:`T_{\text{sig}}` is the signal temperature
- :math:`T_{\text{sys}}` is the system temperature
- :math:`B` is the bandwidth
- :math:`t` is the integration time

For a 5σ detection, we solve for the required integration time:

.. math::

   t = \left(\frac{5 \cdot T_{\text{sys}}}{T_{\text{sig}} \sqrt{B}}\right)^2