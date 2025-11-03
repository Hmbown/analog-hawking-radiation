"""
Validation Framework for Analog Hawking Radiation Experiments

Provides automated validation against known benchmarks, cross-phase consistency
checking, statistical significance validation, and physics model validation.
"""

from .benchmark_validator import BenchmarkValidator
from .cross_phase_validator import CrossPhaseValidator
from .physics_model_validator import PhysicsModelValidator
from .quality_assurance import QualityAssuranceSystem
from .validation_framework import ValidationFramework, ValidationResult

__all__ = [
    'QualityAssuranceSystem',
    'ValidationFramework', 
    'ValidationResult',
    'BenchmarkValidator',
    'CrossPhaseValidator',
    'PhysicsModelValidator'
]