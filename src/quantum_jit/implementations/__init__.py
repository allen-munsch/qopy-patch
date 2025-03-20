"""
Quantum implementations for various patterns.
"""
from quantum_jit.implementations.matrix_multiply import create_quantum_matrix_multiply
from quantum_jit.implementations.fourier_transform import create_quantum_fourier_transform
from quantum_jit.implementations.search import create_quantum_search
from quantum_jit.implementations.optimization import create_quantum_optimization
from quantum_jit.implementations.binary_function import create_quantum_binary_evaluation
from quantum_jit.implementations.selector import create_quantum_implementation

__all__ = [
    'create_quantum_implementation',
    'create_quantum_matrix_multiply',
    'create_quantum_fourier_transform',
    'create_quantum_search',
    'create_quantum_optimization',
    'create_quantum_binary_evaluation'
]