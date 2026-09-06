"""Reproducible Vietnamese experiment pipeline for the Meta-Judge project.

The package intentionally keeps data loading, preprocessing, generation,
metric scoring, and correlation as separate layers.  The original author's
modules under ``generation/`` and ``metrics/`` remain available and are used
by the metric adapter when their dependencies are installed.
"""

__version__ = "0.1.0"
