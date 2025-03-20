"""
Decision logic for when to use quantum implementations.
"""
from typing import Dict, Any, Callable, Tuple


def should_use_quantum(func_id: int, 
                      performance_data: Dict[int, Dict],
                      quantum_implementations: Dict[int, Callable],
                      min_speedup: float = 1.1) -> bool:
    """
    Determine if quantum implementation should be used.
    
    Args:
        func_id: Function ID
        performance_data: Performance data dictionary
        quantum_implementations: Dictionary of quantum implementations
        min_speedup: Minimum speedup required to use quantum version
        
    Returns:
        True if quantum implementation should be used
    """
    if func_id not in quantum_implementations:
        return False
    
    if func_id not in performance_data:
        return False
    
    perf = performance_data[func_id]
    
    # Only use quantum if it's correct and faster than minimum speedup
    return perf['correct'] and perf['speedup'] >= min_speedup


def compare_results(result1: Any, result2: Any, rtol: float = 1e-2, atol: float = 1e-2) -> bool:
    """
    Compare two results for approximate equality.
    
    Args:
        result1: First result
        result2: Second result
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if results are approximately equal
    """
    import numpy as np
    
    try:
        if isinstance(result1, np.ndarray) and isinstance(result2, np.ndarray):
            return np.allclose(result1, result2, rtol=rtol, atol=atol)
        elif isinstance(result1, dict) and isinstance(result2, dict):
            if set(result1.keys()) != set(result2.keys()):
                return False
            return all(abs(result1[k] - result2[k]) < atol 
                      for k in result1 if isinstance(result1[k], (int, float)))
        else:
            return result1 == result2
    except:
        return False