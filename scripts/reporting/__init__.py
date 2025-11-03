"""
Reporting Module for Analog Hawking Radiation Experiments

Provides comprehensive reporting, visualization, synthesis, and publication
capabilities for multi-phase experimental orchestration.
"""

from .publication_formatter import (
    DataTables,
    LaTeXDocument,
    MarkdownDocument,
    PresentationSlides,
    PublicationFormatter,
)
from .report_generator import ExecutiveSummary, ReportGenerator, ScientificReport, TechnicalReport
from .synthesis_engine import MetaAnalysis, PatternRecognition, SynthesisEngine, TrendAnalysis
from .visualization_pipeline import FigureSpecification, VisualizationBundle, VisualizationPipeline

__all__ = [
    # Main classes
    'ReportGenerator',
    'VisualizationPipeline', 
    'SynthesisEngine',
    'PublicationFormatter',
    
    # Data structures
    'ScientificReport',
    'ExecutiveSummary',
    'TechnicalReport',
    'FigureSpecification',
    'VisualizationBundle',
    'TrendAnalysis',
    'PatternRecognition', 
    'MetaAnalysis',
    'LaTeXDocument',
    'MarkdownDocument',
    'PresentationSlides',
    'DataTables'
]

__version__ = "1.0.0"
__author__ = "Analog Hawking Radiation Research Team"