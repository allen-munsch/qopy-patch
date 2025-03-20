"""
Pattern detection for quantum-accelerable code patterns.

This package provides tools to identify and analyze code patterns
that could benefit from quantum acceleration.
"""

from quantum_jit.patterns.core import analyze_function, analyze_source
from quantum_jit.patterns.detectors import (
    # Import all detectors for easy access
    detect_matrix_multiply,
    detect_fourier_transform,
    detect_search,
    detect_optimization,
    detect_quantum_simulation,
    detect_machine_learning,
    detect_graph_algorithm,
    detect_encryption,
    detect_sorting,
    detect_sampling
)

# Dictionary of all available detectors
AVAILABLE_DETECTORS = {
    "matrix_multiplication": detect_matrix_multiply,
    "fourier_transform": detect_fourier_transform,
    "search_algorithm": detect_search,
    "optimization": detect_optimization,
    "quantum_simulation": detect_quantum_simulation,
    "machine_learning": detect_machine_learning,
    "graph_algorithm": detect_graph_algorithm,
    "encryption": detect_encryption,
    "sorting": detect_sorting,
    "sampling": detect_sampling
}

__all__ = [
    'analyze_function', 
    'analyze_source',
    'AVAILABLE_DETECTORS',
    'detect_matrix_multiply',
    'detect_fourier_transform',
    'detect_search',
    'detect_optimization',
    'detect_quantum_simulation',
    'detect_machine_learning',
    'detect_graph_algorithm',
    'detect_encryption',
    'detect_sorting',
    'detect_sampling'
]