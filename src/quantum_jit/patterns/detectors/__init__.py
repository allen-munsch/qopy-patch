"""
Pattern detector implementations for quantum-accelerable code patterns.

This package provides detector functions for various algorithm and computation patterns
that could benefit from quantum acceleration.
"""

# Import all detectors from the modules
from quantum_jit.patterns.detectors.basic import (
    detect_matrix_multiply,
    detect_fourier_transform,
    detect_search
)

from quantum_jit.patterns.detectors.optimization import (
    detect_optimization
)

from quantum_jit.patterns.detectors.simulation import (
    detect_quantum_simulation
)

from quantum_jit.patterns.detectors.machine_learning import (
    detect_machine_learning
)

from quantum_jit.patterns.detectors.graph import (
    detect_graph_algorithm
)

from quantum_jit.patterns.detectors.cryptography import (
    detect_encryption
)

from quantum_jit.patterns.detectors.algorithms import (
    detect_sorting,
    detect_sampling
)

__all__ = [
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