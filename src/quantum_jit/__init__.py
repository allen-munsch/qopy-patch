"""
Quantum Copy-and-Patch JIT
=========================

A library that automatically identifies and accelerates classical code with quantum computing.

Features:
- Pattern detection to identify code suitable for quantum acceleration
- Circuit generation optimized for detected patterns
- Runtime optimization with circuit caching and batching
- Performance-based decision making for when to use quantum acceleration

Example usage:
    ```python
    from quantum_jit import qjit
    
    @qjit
    def hadamard_transform(input_vector):
        # Classical implementation
        n = len(input_vector)
        h_matrix = np.ones((n, n))
        
        for i in range(n):
            for j in range(n):
                if bin(i & j).count('1') % 2 == 1:
                    h_matrix[i, j] = -1
        
        h_matrix = h_matrix / np.sqrt(n)
        return np.dot(h_matrix, input_vector)
    
    # First call: benchmarks classical vs quantum
    result = hadamard_transform(np.array([1, 0, 0, 0]))
    
    # Subsequent calls: may use quantum if beneficial
    result = hadamard_transform(np.array([1, 0, 0, 0]))
    ```
"""

# Version
__version__ = "0.1.0"

# Core exports
from quantum_jit.core import qjit, QuantumJITCompiler

# Export pattern detection for advanced usage
from quantum_jit.pattern_detection.base_detector import BasePatternDetector
from quantum_jit.pattern_detection.common_patterns import (
    detect_matrix_multiply,
    detect_fourier_transform, 
    detect_search
)
from quantum_jit.pattern_detection.function_analyzer import FunctionAnalyzer

# Export circuit generation for advanced usage
from quantum_jit.circuit_generation.circuit_generator import QuantumCircuitGenerator
from quantum_jit.circuit_generation.circuit_optimizer import CircuitOptimizer

# Export runtime components for advanced usage
from quantum_jit.runtime.circuit_cache import CircuitCache
from quantum_jit.runtime.execution_manager import ExecutionManager
from quantum_jit.runtime.result_processor import ResultProcessor

# Define what's available with from quantum_jit import *
__all__ = [
    'qjit',
    'QuantumJITCompiler',
    'BasePatternDetector',
    'FunctionAnalyzer',
    'QuantumCircuitGenerator',
    'CircuitOptimizer',
    'CircuitCache',
    'ExecutionManager',
    'ResultProcessor'
]
