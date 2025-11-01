#!/usr/bin/env python3
"""
Graphics control utilities for Analog Hawking Radiation Analysis.

This module provides utilities to control graphics generation across the codebase,
allowing for headless operation in CI/CD environments and user-configurable
graphics output.
"""

from __future__ import annotations

import os
import argparse
from typing import Optional, Any
from contextlib import contextmanager


def should_generate_plots() -> bool:
    """
    Check if plots should be generated based on environment variables.

    Returns:
        bool: True if plots should be generated, False if they should be skipped
    """
    # Environment variable takes precedence
    no_plots_env = os.getenv("ANALOG_HAWKING_NO_PLOTS")
    if no_plots_env is not None:
        return no_plots_env.lower() not in ("1", "true", "yes", "on")

    # Default to generating plots
    return True


def add_graphics_argument(parser: argparse.ArgumentParser,
                         help_text: Optional[str] = None) -> argparse.ArgumentParser:
    """
    Add graphics control argument to an argument parser.

    Args:
        parser: The argument parser to modify
        help_text: Custom help text for the argument

    Returns:
        The modified argument parser for chaining
    """
    default_help = "Skip graphics generation (faster for CI/CD and batch processing)"
    help_text = help_text or default_help

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help=help_text,
        default=False
    )

    # Also add the inverse flag for clarity
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Force plot generation even if ANALOG_HAWKING_NO_PLOTS is set",
        default=False
    )

    return parser


def get_graphics_preference(args: Optional[argparse.Namespace] = None) -> bool:
    """
    Determine whether to generate graphics based on arguments and environment.

    Args:
        args: Parsed command line arguments (optional)

    Returns:
        bool: True if plots should be generated, False otherwise
    """
    # Command line arguments take precedence over environment
    if args is not None:
        if hasattr(args, 'generate_plots') and args.generate_plots:
            return True
        if hasattr(args, 'no_plots') and args.no_plots:
            return False

    # Fall back to environment variable
    return should_generate_plots()


@contextmanager
def configure_matplotlib():
    """
    Context manager for configuring matplotlib for headless operation.

    This should be used around plotting code to ensure proper configuration
    for headless environments when graphics are disabled.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    # Save original backend
    original_backend = matplotlib.get_backend()

    try:
        if not should_generate_plots():
            # Use Agg backend for headless operation
            matplotlib.use('Agg', force=True)
        yield
    finally:
        # Restore original backend
        if not should_generate_plots():
            matplotlib.use(original_backend, force=True)

        # Clean up any open figures to prevent memory leaks
        plt.close('all')


def conditional_savefig(filename: str, *args, **kwargs) -> bool:
    """
    Conditionally save a figure based on graphics preferences.

    Args:
        filename: Output filename for the figure
        *args: Arguments to pass to plt.savefig
        **kwargs: Keyword arguments to pass to plt.savefig

    Returns:
        bool: True if the figure was saved, False if skipped
    """
    if not should_generate_plots():
        return False

    import matplotlib.pyplot as plt
    plt.savefig(filename, *args, **kwargs)
    return True


def skip_plotting_message(operation: str = "plotting") -> None:
    """
    Print a message indicating that plotting is being skipped.

    Args:
        operation: Description of the plotting operation being skipped
    """
    print(f"Skipping {operation} (graphics generation disabled)")


class GraphicsController:
    """
    A class to manage graphics generation settings across a script.

    This provides a more object-oriented approach to graphics control,
    useful for complex scripts with multiple plotting phases.
    """

    def __init__(self, enable_plots: Optional[bool] = None,
                 verbose: bool = True):
        """
        Initialize the graphics controller.

        Args:
            enable_plots: Force enable/disable plots. If None, uses environment/args
            verbose: Whether to print messages when skipping plots
        """
        self.enable_plots = enable_plots if enable_plots is not None else should_generate_plots()
        self.verbose = verbose

    def should_plot(self) -> bool:
        """Check if plotting should be enabled."""
        return self.enable_plots

    def save_figure(self, filename: str, *args, **kwargs) -> bool:
        """
        Save a figure if plotting is enabled.

        Args:
            filename: Output filename
            *args: Arguments to pass to plt.savefig
            **kwargs: Keyword arguments to pass to plt.savefig

        Returns:
            True if figure was saved, False if skipped
        """
        if not self.enable_plots:
            if self.verbose:
                skip_plotting_message(f"saving figure to {filename}")
            return False

        import matplotlib.pyplot as plt
        plt.savefig(filename, *args, **kwargs)
        return True

    def configure_matplotlib(self):
        """Configure matplotlib for the current graphics settings."""
        if not self.enable_plots:
            import matplotlib
            matplotlib.use('Agg', force=True)

    def __enter__(self):
        """Context manager entry."""
        self.configure_matplotlib()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up any open figures."""
        if not self.enable_plots:
            import matplotlib.pyplot as plt
            plt.close('all')


# Convenience function for backward compatibility
def get_graphics_controller(enable_plots: Optional[bool] = None,
                          verbose: bool = True) -> GraphicsController:
    """
    Get a graphics controller instance.

    Args:
        enable_plots: Force enable/disable plots
        verbose: Whether to print skip messages

    Returns:
        GraphicsController instance
    """
    return GraphicsController(enable_plots=enable_plots, verbose=verbose)