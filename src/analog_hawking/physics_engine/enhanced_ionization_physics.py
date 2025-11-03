"""
Enhanced Ionization Physics for Analog Hawking Radiation Analysis

This module implements comprehensive ionization models for high-intensity laser-plasma
interactions at ELI facility conditions. It addresses the critical missing ionization
physics identified in the scientific review:

1. ADK (Ammosov-Delone-Krainov) tunneling ionization
2. PPT (Perelomov-Popov-Terent'ev) ionization theory
3. Multi-ionization dynamics for different target materials
4. Collisional ionization and recombination processes
5. Ionization front propagation dynamics

Author: Enhanced Physics Implementation
Date: November 2025
References:
- ADK: Ammosov, Delone, Krainov (1986) Sov. Phys. JETP 64, 1191
- PPT: Perelomov, Popov, Terent'ev (1966) Sov. Phys. JETP 23, 924
- Yudin & Ivanov (2001) Phys. Rep. 357, 119
- ELI Beamlines Ionization Physics Documentation

NOTE: Placeholder formulas; use for exploratory prototyping only until constants
and scaling factors are benchmarked against trusted references.
"""

import warnings
from typing import Callable, Dict, List

import numpy as np
from scipy.constants import c, e, hbar, k, m_e, pi
from scipy.integrate import odeint
from scipy.special import gammaln

LOG_MIN_RATE = np.log(np.finfo(float).tiny)
LOG_MAX_RATE = np.log(np.finfo(float).max)


class AtomicSpecies:
    """Atomic data for different ionization states"""

    def __init__(self, Z: int, name: str, ionization_potentials: List[float]):
        """
        Initialize atomic species data

        Args:
            Z: Atomic number
            name: Element name
            ionization_potentials: List of ionization potentials in eV
        """
        self.Z = Z
        self.name = name
        self.Ip = np.array(ionization_potentials) * e  # Convert to Joules
        self.n_states = len(ionization_potentials)

    def get_ionization_potential(self, charge_state: int) -> float:
        """Get ionization potential for given charge state"""
        if 0 <= charge_state < self.n_states:
            return self.Ip[charge_state]
        else:
            return float("inf")


# Common atomic species for ELI experiments
ATOMIC_DATA = {
    "H": AtomicSpecies(1, "Hydrogen", [13.6]),
    "He": AtomicSpecies(2, "Helium", [24.6, 54.4]),
    "C": AtomicSpecies(6, "Carbon", [11.3, 24.4, 47.9, 64.5, 392.1, 490.0]),
    "Al": AtomicSpecies(
        13,
        "Aluminum",
        [5.99, 18.8, 28.4, 120.0, 153.8, 190.5, 241.8, 284.6, 330.2, 398.8, 442.1, 2086.0, 2304.0],
    ),
    "Si": AtomicSpecies(
        14,
        "Silicon",
        [
            8.15,
            16.3,
            33.5,
            45.1,
            67.0,
            99.2,
            151.1,
            181.7,
            205.3,
            236.6,
            274.7,
            330.4,
            400.7,
            476.2,
        ],
    ),
    "Au": AtomicSpecies(79, "Gold", [9.22, 20.5]),  # Simplified - only first few states
}


class ADKIonizationModel:
    """
    ADK (Ammosov-Delone-Krainov) tunneling ionization model for strong fields
    """

    def __init__(self, atomic_species: AtomicSpecies):
        """
        Initialize ADK model for specific atomic species

        Args:
            atomic_species: AtomicSpecies object containing ionization potentials
        """
        self.atom = atomic_species
        self.E_a = (2 * self.atom.Ip) ** 1.5 / (e * hbar)  # Atomic field strength
        warnings.warn(
            "ADK ionization model constants are placeholders; calibrate against benchmarks "
            "before relying on absolute rates.",
            RuntimeWarning,
            stacklevel=2,
        )

    def log_adk_rate(self, E_field: float, charge_state: int) -> float:
        """
        Calculate natural logarithm of the ADK ionization rate.

        Working in log-space avoids catastrophic underflow for strong-field regimes,
        while preserving monotonic scaling information.
        """
        if charge_state >= self.atom.n_states or E_field <= 0:
            return float("-inf")

        Ip = self.atom.get_ionization_potential(charge_state)
        n_eff = self.atom.Z * np.sqrt(Ip / (13.6 * e))
        l_eff = n_eff - 1
        kappa = np.sqrt(2 * m_e * Ip) / hbar
        E_a_state = self.E_a[charge_state]

        log_prefactor = np.log(E_a_state) - np.log(2 * E_field)
        log_kappa_term = np.log(2 * kappa**3 / (pi * E_field))
        log_power = (2 * n_eff - 1) * np.log(2 * E_field / E_a_state)
        exponent_term = -2 * kappa**3 / (3 * E_field)

        log_W_ADK = log_prefactor + log_kappa_term + log_power + exponent_term
        log_C_n2 = (
            2 * n_eff * np.log(2)
            - np.log(n_eff)
            - gammaln(n_eff + l_eff + 1)
            - gammaln(n_eff - l_eff)
        )

        return float(log_W_ADK + log_C_n2)

    def adk_rate(self, E_field: float, charge_state: int) -> float:
        """
        Calculate ADK ionization rate. For diagnostics requiring absolute rates,
        values are exponentiated from log-space and clipped to double precision
        bounds to avoid underflow/overflow.
        """
        log_rate = self.log_adk_rate(E_field, charge_state)
        if not np.isfinite(log_rate):
            return 0.0

        clipped_log = np.clip(log_rate, LOG_MIN_RATE, LOG_MAX_RATE)
        return float(np.exp(clipped_log))

    def adk_rates_vectorized(self, E_field: np.ndarray, charge_state: int) -> np.ndarray:
        """Vectorized ADK rate calculation"""
        rates = np.zeros_like(E_field)
        for i, E in enumerate(E_field):
            rates[i] = self.adk_rate(E, charge_state)
        return rates


class PPTIonizationModel:
    """
    PPT (Perelomov-Popov-Terent'ev) ionization model - generalization of ADK
    """

    def __init__(self, atomic_species: AtomicSpecies):
        """
        Initialize PPT model for specific atomic species

        Args:
            atomic_species: AtomicSpecies object containing ionization potentials
        """
        self.atom = atomic_species
        warnings.warn(
            "PPT ionization rates here are simplified; validate scaling factors before use.",
            RuntimeWarning,
            stacklevel=2,
        )

    def ppt_rate(self, E_field: float, charge_state: int, omega: float) -> float:
        """
        Calculate PPT ionization rate for arbitrary frequency

        Args:
            E_field: Electric field amplitude in V/m
            charge_state: Current charge state
            omega: Laser frequency in rad/s

        Returns:
            Ionization rate in s^-1
        """
        if charge_state >= self.atom.n_states:
            return 0.0

        Ip = self.atom.get_ionization_potential(charge_state)
        kappa = np.sqrt(2 * m_e * Ip) / hbar

        # PPT parameters
        F0 = E_field
        gamma_K = omega * kappa * m_e / (e * F0)  # Keldysh parameter

        if gamma_K < 1:  # Tunneling regime
            # Use ADK limit
            adk_model = ADKIonizationModel(self.atom)
            return adk_model.adk_rate(E_field, charge_state)
        else:  # Multiphoton regime
            # Multiphoton absorption rate
            n_photons = int(np.ceil(Ip / (hbar * omega)))

            # Simplified multiphoton rate
            W_MP = (
                (F0 / (2 * m_e * omega)) ** (2 * n_photons)
                * (omega / (2 * pi))
                * np.exp(-2 * n_photons * np.log(1 + gamma_K**2))
            )

            return W_MP

    def keldysh_parameter(self, E_field: float, charge_state: int, omega: float) -> float:
        """
        Calculate Keldysh parameter

        Args:
            E_field: Electric field strength in V/m
            charge_state: Current charge state
            omega: Laser frequency in rad/s

        Returns:
            Keldysh parameter (dimensionless)
        """
        Ip = self.atom.get_ionization_potential(charge_state)
        kappa = np.sqrt(2 * m_e * Ip) / hbar

        gamma_K = omega * kappa * m_e / (e * E_field)
        return gamma_K


class CollisionalIonizationModel:
    """
    Collisional ionization model for electron impact ionization
    """

    def __init__(self, atomic_species: AtomicSpecies):
        """
        Initialize collisional ionization model

        Args:
            atomic_species: AtomicSpecies object containing ionization potentials
        """
        self.atom = atomic_species

    def lotz_cross_section(self, electron_energy: float, charge_state: int) -> float:
        """
        Lotz formula for electron impact ionization cross-section

        Args:
            electron_energy: Electron energy in Joules
            charge_state: Current charge state

        Returns:
            Cross-section in m^2
        """
        if charge_state >= self.atom.n_states:
            return 0.0

        Ip = self.atom.get_ionization_potential(charge_state)
        E_eV = electron_energy / e
        Ip_eV = Ip / e

        if E_eV <= Ip_eV:
            return 0.0

        # Lotz formula (simplified)
        # σ = A * ln(E/Ip) / (E * Ip) * (1 - B * exp(-C * E/Ip))
        A = 4e-20  # Typical value in m^2
        B = 0.5
        C = 1.0

        sigma = A * np.log(E_eV / Ip_eV) / (E_eV * Ip_eV) * (1 - B * np.exp(-C * E_eV / Ip_eV))

        return sigma

    def collisional_rate(
        self, electron_density: float, electron_temperature: float, charge_state: int
    ) -> float:
        """
        Calculate collisional ionization rate

        Args:
            electron_density: Electron density in m^-3
            electron_temperature: Electron temperature in Joules
            charge_state: Current charge state

        Returns:
            Collisional ionization rate in s^-1
        """
        # Maxwellian-averaged collisional rate
        # <σv> ~ ∫ σ(v) * v * f(v) dv

        # Simplified using thermal velocity
        v_th = np.sqrt(8 * electron_temperature / (pi * m_e))

        # Average cross-section at thermal energy
        sigma_avg = self.lotz_cross_section(2 * electron_temperature, charge_state)

        rate = electron_density * sigma_avg * v_th
        return rate


class RecombinationModel:
    """
    Recombination processes (radiative and three-body recombination)
    """

    def __init__(self, atomic_species: AtomicSpecies):
        """
        Initialize recombination model

        Args:
            atomic_species: AtomicSpecies object containing ionization potentials
        """
        self.atom = atomic_species

    def radiative_recombination_rate(
        self, electron_density: float, electron_temperature: float, charge_state: int
    ) -> float:
        """
        Calculate radiative recombination rate

        Args:
            electron_density: Electron density in m^-3
            electron_temperature: Electron temperature in Joules
            charge_state: Current charge state

        Returns:
            Radiative recombination rate in s^-1
        """
        if charge_state <= 0:
            return 0.0

        # Simplified radiative recombination rate
        # α_rr ∝ Z^2 / T^0.5
        alpha_0 = 2.6e-19  # Reference rate in m^3/s
        Z_eff = charge_state

        alpha_rr = alpha_0 * Z_eff**2 * np.sqrt(13.6 * e / electron_temperature)

        rate = electron_density * alpha_rr
        return rate

    def three_body_recombination_rate(
        self, electron_density: float, electron_temperature: float, charge_state: int
    ) -> float:
        """
        Calculate three-body recombination rate

        Args:
            electron_density: Electron density in m^-3
            electron_temperature: Electron temperature in Joules
            charge_state: Current charge state

        Returns:
            Three-body recombination rate in s^-1
        """
        if charge_state <= 0:
            return 0.0

        Ip = self.atom.get_ionization_potential(charge_state - 1)

        # Three-body recombination rate coefficient
        # α_3b ∝ n_e * Z^6 / T^9/2 * exp(Ip/kT)
        alpha_0 = 1e-39  # Reference rate in m^6/s
        Z_eff = charge_state

        try:
            alpha_3b = (
                alpha_0
                * Z_eff**6
                * (13.6 * e / electron_temperature) ** (9 / 2)
                * np.exp(Ip / (k * electron_temperature))
            )
        except (OverflowError, FloatingPointError):
            alpha_3b = 0.0

        rate = electron_density**2 * alpha_3b
        return rate


class IonizationDynamics:
    """
    Complete ionization dynamics simulation including all processes
    """

    def __init__(self, atomic_species: AtomicSpecies, laser_wavelength: float = 800e-9):
        """
        Initialize ionization dynamics model

        Args:
            atomic_species: AtomicSpecies object
            laser_wavelength: Laser wavelength in meters
        """
        self.atom = atomic_species
        self.omega_l = 2 * pi * c / laser_wavelength

        # Initialize sub-models
        self.adk_model = ADKIonizationModel(atomic_species)
        self.ppt_model = PPTIonizationModel(atomic_species)
        self.collisional_model = CollisionalIonizationModel(atomic_species)
        self.recombination_model = RecombinationModel(atomic_species)

    def rate_equations(
        self,
        y: np.ndarray,
        t: np.ndarray,
        E_field_func: Callable,
        n_e_func: Callable,
        T_e_func: Callable,
    ) -> np.ndarray:
        """
        Rate equations for ionization state populations

        Args:
            y: Array of ionization state populations
            t: Time array
            E_field_func: Function returning electric field at time t
            n_e_func: Function returning electron density at time t
            T_e_func: Function returning electron temperature at time t

        Returns:
            Time derivatives of ionization populations
        """
        dydt = np.zeros_like(y)

        # Get current conditions
        E_field = E_field_func(t)
        n_e = n_e_func(t)
        T_e = T_e_func(t)

        # Calculate rates for each ionization state
        for charge_state in range(self.atom.n_states):
            if charge_state == 0:
                # Neutral atom
                if self.atom.n_states > 0:
                    # Ionization from neutral to +1
                    W_field = self.ppt_model.ppt_rate(abs(E_field), charge_state, self.omega_l)
                    W_coll = self.collisional_model.collisional_rate(n_e, T_e, charge_state)

                    # Recombination to neutral
                    if n_e > 0:
                        W_recomb_rad = self.recombination_model.radiative_recombination_rate(
                            n_e, T_e, charge_state + 1
                        )
                        W_recomb_3b = self.recombination_model.three_body_recombination_rate(
                            n_e, T_e, charge_state + 1
                        )
                        W_recomb = W_recomb_rad + W_recomb_3b
                    else:
                        W_recomb = 0.0

                    dydt[0] = -y[0] * (W_field + W_coll) + y[1] * W_recomb

            elif charge_state < self.atom.n_states - 1:
                # Intermediate ionization states
                # Ionization to higher charge state
                W_field = self.ppt_model.ppt_rate(abs(E_field), charge_state, self.omega_l)
                W_coll = self.collisional_model.collisional_rate(n_e, T_e, charge_state)

                # Recombination from higher charge state
                if n_e > 0:
                    W_recomb_rad = self.recombination_model.radiative_recombination_rate(
                        n_e, T_e, charge_state + 1
                    )
                    W_recomb_3b = self.recombination_model.three_body_recombination_rate(
                        n_e, T_e, charge_state + 1
                    )
                    W_recomb = W_recomb_rad + W_recomb_3b
                else:
                    W_recomb = 0.0

                # Recombination to lower charge state
                if n_e > 0:
                    W_recomb_lower_rad = self.recombination_model.radiative_recombination_rate(
                        n_e, T_e, charge_state
                    )
                    W_recomb_lower_3b = self.recombination_model.three_body_recombination_rate(
                        n_e, T_e, charge_state
                    )
                    W_recomb_lower = W_recomb_lower_rad + W_recomb_lower_3b
                else:
                    W_recomb_lower = 0.0

                dydt[charge_state] = (
                    y[charge_state - 1] * (W_field + W_coll)
                    - y[charge_state] * (W_field + W_coll + W_recomb)
                    + y[charge_state + 1] * W_recomb
                    - y[charge_state] * W_recomb_lower
                )

            else:
                # Highest ionization state
                # Can only recombine
                if n_e > 0:
                    W_recomb_rad = self.recombination_model.radiative_recombination_rate(
                        n_e, T_e, charge_state
                    )
                    W_recomb_3b = self.recombination_model.three_body_recombination_rate(
                        n_e, T_e, charge_state
                    )
                    W_recomb = W_recomb_rad + W_recomb_3b
                else:
                    W_recomb = 0.0

                dydt[charge_state] = (
                    y[charge_state - 1]
                    * (
                        self.ppt_model.ppt_rate(abs(E_field), charge_state - 1, self.omega_l)
                        + self.collisional_model.collisional_rate(n_e, T_e, charge_state - 1)
                    )
                    - y[charge_state] * W_recomb
                )

        return dydt

    def simulate_ionization(
        self,
        initial_density: float,
        time_array: np.ndarray,
        E_field_func: Callable,
        n_e_func: Callable,
        T_e_func: Callable,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate ionization dynamics over time

        Args:
            initial_density: Initial neutral density in m^-3
            time_array: Time array for simulation
            E_field_func: Function returning electric field at time t
            n_e_func: Function returning electron density at time t
            T_e_func: Function returning electron temperature at time t

        Returns:
            Dictionary with ionization evolution data
        """
        # Initial conditions (all neutral)
        y0 = np.zeros(self.atom.n_states)
        y0[0] = initial_density

        # Solve rate equations
        solution = odeint(
            self.rate_equations, y0, time_array, args=(E_field_func, n_e_func, T_e_func)
        )

        # Calculate derived quantities
        total_density = np.sum(solution, axis=1)
        charge_state_distribution = solution / total_density[:, np.newaxis]
        average_charge_state = np.sum(
            charge_state_distribution * np.arange(self.atom.n_states), axis=1
        )

        return {
            "time": time_array,
            "populations": solution,
            "charge_state_distribution": charge_state_distribution,
            "average_charge_state": average_charge_state,
            "total_density": total_density,
        }

    def ionization_front_position(
        self, time_array: np.ndarray, E_field_profile: Callable, intensity: float
    ) -> np.ndarray:
        """
        Calculate ionization front position (simplified model)

        Args:
            time_array: Time array
            E_field_profile: Function for spatial field profile
            intensity: Laser intensity in W/m^2

        Returns:
            Ionization front position over time
        """
        # Simplified ionization front model
        # Front moves with group velocity modified by ionization
        v_group = c * np.sqrt(1 - self.omega_pe**2 / self.omega_l**2)

        front_position = np.zeros_like(time_array)
        for i, t in enumerate(time_array):
            # Simple model: front moves at modified group velocity
            front_position[i] = v_group * t * (1 - 0.1 * np.sin(self.omega_l * t))

        return front_position


def test_ionization_physics():
    """
    Test the ionization physics implementation
    """
    print("Testing Enhanced Ionization Physics")
    print("=" * 50)

    # Test with Aluminum (common ELI target)
    atom = ATOMIC_DATA["Al"]
    print(f"Testing with {atom.name} (Z={atom.Z})")
    print(f"Number of ionization states: {atom.n_states}")

    # Create ionization dynamics model
    ionization = IonizationDynamics(atom, laser_wavelength=800e-9)

    # Test ADK rates
    E_fields = np.logspace(10, 14, 10)  # V/m
    print("\nADK Ionization Rates (neutral -> +1):")
    for E in E_fields[::3]:  # Print every 3rd value
        rate = ionization.adk_model.adk_rate(E, 0)
        print(f"  E = {E:.2e} V/m: W_ADK = {rate:.2e} s^-1")

    # Test PPT rates with different Keldysh parameters
    print("\nKeldysh Parameters and PPT Rates:")
    omega_l = 2 * pi * c / 800e-9
    for E in [1e11, 1e12, 1e13, 1e14]:
        gamma_K = ionization.ppt_model.keldysh_parameter(E, 0, omega_l)
        ppt_rate = ionization.ppt_model.ppt_rate(E, 0, omega_l)
        regime = "Tunneling" if gamma_K < 1 else "Multiphoton"
        print(f"  E = {E:.1e} V/m: γ_K = {gamma_K:.2f} ({regime}), W_PPT = {ppt_rate:.2e} s^-1")

    # Test collisional ionization
    print("\nCollisional Ionization Rates:")
    n_e = 1e19  # m^-3
    T_e = 1e6 * e  # 1 MK in Joules
    for charge_state in range(3):
        rate = ionization.collisional_model.collisional_rate(n_e, T_e, charge_state)
        print(f"  Charge state +{charge_state}: W_coll = {rate:.2e} s^-1")

    # Test recombination
    print("\nRecombination Rates:")
    for charge_state in [1, 2, 3]:
        rad_rate = ionization.recombination_model.radiative_recombination_rate(
            n_e, T_e, charge_state
        )
        three_body_rate = ionization.recombination_model.three_body_recombination_rate(
            n_e, T_e, charge_state
        )
        print(
            f"  +{charge_state} -> +{charge_state-1}: Rad = {rad_rate:.2e}, 3-body = {three_body_rate:.2e} s^-1"
        )

    print("\nIonization Physics Test Complete!")


if __name__ == "__main__":
    test_ionization_physics()
