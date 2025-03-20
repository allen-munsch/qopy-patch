"""
Benchmarking utilities for quantum JIT system.
"""
import time
from typing import Dict, Any, Callable, Tuple, Optional


def time_execution(func: Callable, args: tuple, kwargs: dict) -> Tuple[Any, float]:
    """
    Time the execution of a function.
    
    Args:
        func: Function to time
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Tuple of (result, execution_time)
    """
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        raise


def print_benchmark_results(
    func_name: str, 
    classical_time: float, 
    quantum_time: float, 
    speedup: float,
    is_correct: bool
) -> None:
    """
    Print benchmark results to console.
    
    Args:
        func_name: Name of the function
        classical_time: Classical execution time
        quantum_time: Quantum execution time
        speedup: Speedup factor
        is_correct: Whether quantum result is correct
    """
    print(f"Function {func_name} benchmarked:")
    print(f"  Classical time: {classical_time:.6f}s")
    print(f"  Quantum time: {quantum_time:.6f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Correct results: {is_correct}")


def benchmark_function(
    classical_func: Callable,
    quantum_func: Optional[Callable], 
    args: tuple, 
    kwargs: dict,
    compare_func: Callable
) -> Dict[str, Any]:
    """
    Benchmark classical vs quantum implementations.
    
    Args:
        classical_func: Classical implementation
        quantum_func: Quantum implementation (or None)
        args: Function arguments
        kwargs: Function keyword arguments
        compare_func: Function to compare results
        
    Returns:
        Dictionary with benchmark results
    """
    # Time classical execution
    classical_result, classical_time = time_execution(classical_func, args, kwargs)
    
    # If no quantum implementation, return early
    if quantum_func is None:
        return {
            'classical_time': classical_time,
            'quantum_time': None,
            'speedup': None,
            'correct': None
        }
    
    # Time quantum execution
    quantum_result, quantum_time = time_execution(quantum_func, args, kwargs)
    
    # Compare results
    is_correct = compare_func(classical_result, quantum_result)
    
    # Calculate speedup
    speedup = classical_time / quantum_time if quantum_time > 0 else 0
    
    return {
        'classical_time': classical_time,
        'quantum_time': quantum_time,
        'speedup': speedup,
        'correct': is_correct
    }