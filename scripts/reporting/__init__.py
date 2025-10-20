"""
Reporting Module for Analog Hawking Radiation Experiments

Provides comprehensive reporting, visualization, synthesis, and publication
capabilities for multi-phase experimental orchestration.
"""

from .report_generator import ReportGenerator, ScientificReport, ExecutiveSummary, TechnicalReport
from .visualization_pipeline import VisualizationPipeline, FigureSpecification, VisualizationBundle
from .synthesis_engine import SynthesisEngine, TrendAnalysis, PatternRecognition, MetaAnalysis
from .publication_formatter import PublicationFormatter, LaTeXDocument, MarkdownDocument, PresentationSlides, DataTables

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