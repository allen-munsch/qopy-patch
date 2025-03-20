"""
Core implementation of the quantum copy-and-patch JIT system.
"""
import functools
import time
from typing import Dict, Any, List, Callable, Tuple, Optional, Union

# Import components
from quantum_jit.circuit_generation.circuit_generator import QuantumCircuitGenerator
from quantum_jit.circuit_generation.circuit_optimizer import CircuitOptimizer
from quantum_jit.runtime.circuit_cache import CircuitCache
from quantum_jit.runtime.execution_manager import ExecutionManager
from quantum_jit.runtime.result_processor import ResultProcessor

# Import patterns
from quantum_jit.patterns import analyze_function, AVAILABLE_DETECTORS

# Import implementation modules
from quantum_jit.implementations.matrix_multiply import create_quantum_matrix_multiply
from quantum_jit.implementations.fourier_transform import create_quantum_fourier_transform
from quantum_jit.implementations.search import create_quantum_search
from quantum_jit.implementations.optimization import create_quantum_optimization
from quantum_jit.implementations.binary_function import create_quantum_binary_evaluation
from quantum_jit.implementations.selector import create_quantum_implementation
from quantum_jit.decision.decision_maker import compare_results


# Helper functions
def time_execution(func: Callable, args: tuple, kwargs: dict) -> Tuple[Any, float]:
    """Time the execution of a function."""
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        raise





def select_quantum_implementation(pattern_name: str, classical_func: Callable, 
                               components: Dict[str, Any]) -> Optional[Callable]:
    """Select appropriate quantum implementation based on pattern."""
    # Special cases based on function name
    if classical_func.__name__ == "evaluate_all":
        return create_quantum_binary_evaluation(
            classical_func,
            components['circuit_generator'],
            components['circuit_optimizer'],
            components['circuit_cache'],
            components['execution_manager'],
            components['result_processor']
        )
    
    # Pattern-based implementation selection
    implementations = {
        "matrix_multiplication": create_quantum_matrix_multiply,
        "fourier_transform": create_quantum_fourier_transform,
        "search_algorithm": create_quantum_search,
        "optimization": create_quantum_optimization
    }
    
    if pattern_name in implementations:
        return implementations[pattern_name](
            classical_func,
            components['circuit_generator'],
            components['circuit_optimizer'],
            components['circuit_cache'],
            components['execution_manager'],
            components['result_processor']
        )
    
    return None


class QuantumJITCompiler:
    """
    Just-In-Time compiler that dynamically replaces classical functions with 
    quantum implementations when beneficial.
    """
    
    def __init__(self, 
                 backend_name: str = 'qasm_simulator', 
                 auto_patch: bool = True,
                 min_speedup: float = 1.1,
                 verbose: bool = True,
                 cache_size: int = 100,
                 detectors: Optional[Dict[str, Callable]] = None):
        """
        Initialize the quantum JIT compiler.
        
        Args:
            backend_name: Name of the quantum backend to use
            auto_patch: Whether to automatically patch functions
            min_speedup: Minimum speedup required to use quantum version
            verbose: Whether to print performance information
            cache_size: Maximum number of circuits to cache
            detectors: Optional dictionary of custom detectors
        """
        # Initialize components
        self.circuit_generator = QuantumCircuitGenerator()
        self.circuit_optimizer = CircuitOptimizer()
        self.circuit_cache = CircuitCache(max_size=cache_size)
        self.execution_manager = ExecutionManager(backend_name=backend_name)
        self.result_processor = ResultProcessor()
        
        # Create components dictionary for easier passing to functions
        self.components = {
            'circuit_generator': self.circuit_generator,
            'circuit_optimizer': self.circuit_optimizer,
            'circuit_cache': self.circuit_cache,
            'execution_manager': self.execution_manager,
            'result_processor': self.result_processor
        }
        
        # Settings
        self.auto_patch = auto_patch
        self.min_speedup = min_speedup
        self.verbose = verbose
        
        # Performance tracking
        self.performance_data = {}
        self.call_counters = {}
        
        # Patched function registry
        self.quantum_implementations = {}
        
        # Initialize pattern detectors
        self.detectors = detectors or AVAILABLE_DETECTORS
    
    def jit(self, func: Callable) -> Callable:
        """
        Decorator to apply quantum JIT compilation to a function.
        
        Args:
            func: Function to apply JIT to
            
        Returns:
            Wrapped function that may use quantum implementation
        """
        # Important: store the original function ID before wrapping
        original_func = func
        original_func_id = id(original_func)
        self.call_counters[original_func_id] = 0
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use the original function ID for tracking
            self.call_counters[original_func_id] += 1
            call_count = self.call_counters[original_func_id]
            
            # First call: always use classical and benchmark
            if call_count == 1:
                # Timing the classical execution
                classical_result, classical_time = time_execution(original_func, args, kwargs)
                
                # Analyze and potentially create quantum version
                if self.auto_patch:
                    quantum_func = self._analyze_and_patch(original_func)
                    
                    # If we created a quantum version, benchmark it
                    if quantum_func:
                        quantum_result, quantum_time = time_execution(quantum_func, args, kwargs)
                        
                        # Compare results for correctness
                        is_correct = compare_results(classical_result, quantum_result)
                        
                        # Calculate speedup
                        speedup = classical_time / quantum_time if quantum_time > 0 else 0
                        
                        # Store performance data using original function ID
                        self.performance_data[original_func_id] = {
                            'classical_time': classical_time,
                            'quantum_time': quantum_time,
                            'speedup': speedup,
                            'correct': is_correct
                        }
                        
                        if self.verbose:
                            print_benchmark_results(original_func.__name__, classical_time, 
                                                 quantum_time, speedup, is_correct)
                
                return classical_result
            
            # Subsequent calls: decide which implementation to use
            use_quantum = self._should_use_quantum(original_func_id)
            
            if use_quantum:
                if self.verbose:
                    print(f"Using quantum implementation for {original_func.__name__}")
                return self.quantum_implementations[original_func_id](*args, **kwargs)
            else:
                return original_func(*args, **kwargs)
        
        # Store the reference to the original function
        wrapper.__wrapped__ = original_func
        
        return wrapper
    
    def _should_use_quantum(self, func_id: int, args=None, kwargs=None) -> bool:
        """
        Determine if quantum implementation should be used.
        
        Args:
            func_id: Function ID
            args: Function arguments (not used, kept for backward compatibility)
            kwargs: Function keyword arguments (not used, kept for backward compatibility)
            
        Returns:
            True if quantum implementation should be used
        """
        if func_id not in self.quantum_implementations:
            return False
        
        if func_id not in self.performance_data:
            return False
        
        perf = self.performance_data[func_id]
        
        # Only use quantum if it's correct and faster than minimum speedup
        return perf['correct'] and perf['speedup'] >= self.min_speedup
    
    # Method added for backwards compatibility with tests
    def _compare_results(self, result1, result2):
        """Wrapper for compare_results for backward compatibility."""
        return compare_results(result1, result2)
    
    def _analyze_and_patch(self, func: Callable) -> Optional[Callable]:
        """Analyze a function and create a quantum implementation if a pattern is detected."""
        # Use the original function, not a wrapper
        if hasattr(func, '__wrapped__'):
            func = func.__wrapped__
            
        func_id = id(func)
        
        try:
            # Detect quantum patterns
            patterns = analyze_function(func, self.detectors)
            
            if not patterns:
                return None
            
            # Get the highest confidence pattern
            pattern_name = max(patterns.items(), key=lambda x: x[1])[0]
            confidence = patterns[pattern_name]
            
            if self.verbose:
                print(f"Detected {pattern_name} pattern in {func.__name__} with confidence {confidence}")
            
            # Create quantum implementation
            quantum_func = create_quantum_implementation(
                pattern_name, 
                func, 
                self.components, 
                self.verbose
            )
            
            if quantum_func:
                self.quantum_implementations[func_id] = quantum_func
            
            return quantum_func
                
        except Exception as e:
            # If there's an error in analysis, log it but don't crash
            if self.verbose:
                print(f"Error analyzing function {func.__name__}: {e}")
                import traceback
                traceback.print_exc()
            
            return None


# Simplified API
def qjit(func=None, *, auto_patch=True, min_speedup=1.1, verbose=True, cache_size=100, detectors=None):
    """
    Decorator to apply quantum JIT compilation to a function.
    
    Args:
        func: Function to apply JIT to
        auto_patch: Whether to automatically patch with quantum implementation
        min_speedup: Minimum speedup required to use quantum version
        verbose: Whether to print performance information
        cache_size: Maximum number of circuits to cache
        detectors: Optional dictionary of custom detectors
        
    Returns:
        Wrapped function that may use quantum implementation
    """
    # Create a custom compiler with specified parameters
    compiler = QuantumJITCompiler(
        auto_patch=auto_patch,
        min_speedup=min_speedup,
        verbose=verbose,
        cache_size=cache_size,
        detectors=detectors
    )
    
    # Handle both @qjit and @qjit(...)
    if func is None:
        return lambda f: compiler.jit(f)
    
    return compiler.jit(func)