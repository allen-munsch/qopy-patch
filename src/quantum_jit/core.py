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

# Import benchmarking utilities
from quantum_jit.benchmarking.benchmarker import time_execution, print_benchmark_results

# Import decision making logic
from quantum_jit.decision.decision_maker import compare_results

# Import implementation selector
from quantum_jit.implementations.selector import create_quantum_implementation

# Global registry of compiler instances
_compiler_instances = []

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
                 detectors: Optional[Dict[str, Callable]] = None,
                 visualize_after: Optional[int] = None):
        """
        Initialize the quantum JIT compiler.
        
        Args:
            backend_name: Name of the quantum backend to use
            auto_patch: Whether to automatically patch functions
            min_speedup: Minimum speedup required to use quantum version
            verbose: Whether to print performance information
            cache_size: Maximum number of circuits to cache
            detectors: Optional dictionary of custom detectors
            visualize_after: Generate visualizations after this many total function calls
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
        self.visualize_after = visualize_after
        
        # Performance tracking
        self.performance_data = {}
        self.call_counters = {}
        self.total_calls = 0
        
        # Patched function registry
        self.quantum_implementations = {}
        
        # Pattern detection data
        self.pattern_data = {}
        
        # Initialize pattern detectors
        self.detectors = detectors or AVAILABLE_DETECTORS
        
        # Register this instance
        _compiler_instances.append(self)
    
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
            self.total_calls += 1
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
                            'timestamp': time.time(),
                            'function_name': original_func.__name__,
                            'classical_time': classical_time,
                            'quantum_time': quantum_time,
                            'speedup': speedup,
                            'correct': is_correct
                        }
                        
                        if self.verbose:
                            print_benchmark_results(original_func.__name__, classical_time, 
                                                 quantum_time, speedup, is_correct)
                
                # Check if we should generate visualizations
                if self.visualize_after and self.total_calls >= self.visualize_after:
                    if self.verbose:
                        print(f"Generating quantum acceleration visualizations after {self.total_calls} calls")
                    self.visualize_after = None  # Reset so we don't keep visualizing
                    self.visualize_acceleration()
                
                return classical_result
            
            # Subsequent calls: decide which implementation to use
            use_quantum = self._should_use_quantum(original_func_id, args, kwargs)
            
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
            
            # Store pattern data for visualization
            self.pattern_data[func_id] = {
                'timestamp': time.time(),
                'function_name': func.__name__,
                'pattern': pattern_name,
                'confidence': confidence,
                'patterns_detected': patterns
            }
            
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
    
    def visualize_acceleration(self, output_dir='./quantum_viz'):
        """
        Generate visualizations of quantum acceleration data.
        
        Args:
            output_dir: Directory to save visualization files
        """
        try:
            from quantum_jit.visualization import (
                call_graph, 
                pattern_dashboard, 
                performance_tracker,
                code_annotator
            )
            
            import os
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Prepare data for visualizations
            pattern_data = []
            performance_history = []
            function_registry = {}
            
            for func_id, perf_data in self.performance_data.items():
                func_name = perf_data.get('function_name', 'unknown')
                function_registry[func_id] = func_name
                
                # Get pattern information if available
                pattern_info = self.pattern_data.get(func_id, {})
                
                if pattern_info:
                    pattern_data.append({
                        'function': func_name,
                        'pattern': pattern_info.get('pattern', 'unknown'),
                        'confidence': pattern_info.get('confidence', 0),
                        'speedup': perf_data.get('speedup', 0),
                        'timestamp': pattern_info.get('timestamp', time.time())
                    })
                    
                    # Add to performance history
                    performance_history.append({
                        'timestamp': perf_data.get('timestamp', time.time()),
                        'function': func_name,
                        'classical_time': perf_data.get('classical_time', 0),
                        'quantum_time': perf_data.get('quantum_time', 0),
                        'speedup': perf_data.get('speedup', 0)
                    })
            
            # Generate visualizations
            if function_registry:
                call_graph.create_call_graph(
                    self.quantum_implementations, 
                    self.performance_data,
                    function_registry,
                    output_dir=output_dir
                )
            
            if pattern_data:
                pattern_dashboard.visualize_patterns(
                    pattern_data,
                    output_dir=output_dir
                )
                
            if performance_history:
                performance_tracker.visualize_performance_timeline(
                    performance_history,
                    output_dir=output_dir
                )
                
            # Also save the data for potential later use
            self._save_visualization_data(output_dir)
            
            if self.verbose:
                print(f"Visualizations saved to {output_dir}")
                
        except ImportError as e:
            if self.verbose:
                print(f"Visualization requires additional dependencies: {e}")
                print("Please install visualization requirements with: pip install quantum-jit[viz]")
    
    def _save_visualization_data(self, output_dir):
        """Save performance and pattern data for external visualization tools."""
        import json
        import os
        
        # Convert to JSON-serializable format
        performance_json = {}
        for func_id, data in self.performance_data.items():
            func_name = data.get('function_name', f'func_{func_id}')
            performance_json[func_name] = {
                'classical_time': data.get('classical_time', 0),
                'quantum_time': data.get('quantum_time', 0),
                'speedup': data.get('speedup', 0),
                'timestamp': data.get('timestamp', 0),
                'correct': data.get('correct', False)
            }
            
        pattern_json = {}
        for func_id, data in self.pattern_data.items():
            func_name = data.get('function_name', f'func_{func_id}')
            pattern_json[func_name] = {
                'pattern': data.get('pattern', 'unknown'),
                'confidence': data.get('confidence', 0),
                'timestamp': data.get('timestamp', 0)
            }
        
        # Save to JSON files
        with open(os.path.join(output_dir, 'performance_data.json'), 'w') as f:
            json.dump(performance_json, f, indent=2)
            
        with open(os.path.join(output_dir, 'pattern_data.json'), 'w') as f:
            json.dump(pattern_json, f, indent=2)


# Simplified API
def qjit(func=None, *, auto_patch=True, min_speedup=1.1, verbose=True, 
         cache_size=100, detectors=None, visualize_after=None):
    """
    Decorator to apply quantum JIT compilation to a function.
    
    Args:
        func: Function to apply JIT to
        auto_patch: Whether to automatically patch with quantum implementation
        min_speedup: Minimum speedup required to use quantum version
        verbose: Whether to print performance information
        cache_size: Maximum number of circuits to cache
        detectors: Optional dictionary of custom detectors
        visualize_after: Generate visualizations after this many total function calls
        
    Returns:
        Wrapped function that may use quantum implementation
    """
    # Create a custom compiler with specified parameters
    compiler = QuantumJITCompiler(
        auto_patch=auto_patch,
        min_speedup=min_speedup,
        verbose=verbose,
        cache_size=cache_size,
        detectors=detectors,
        visualize_after=visualize_after
    )
    
    # Handle both @qjit and @qjit(...)
    if func is None:
        return lambda f: compiler.jit(f)
    
    return compiler.jit(func)


def visualize_all(output_dir='./quantum_viz'):
    """Generate visualizations for all quantum JIT compilers."""
    for compiler in _compiler_instances:
        compiler.visualize_acceleration(output_dir)