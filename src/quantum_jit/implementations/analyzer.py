"""
Analyze functions to detect quantum-accelerable patterns.
"""
from typing import Callable, Dict, Any, Optional

from quantum_jit.patterns import analyze_function
from quantum_jit.implementations.selector import create_quantum_implementation


def analyze_and_patch(
    func: Callable,
    components: Dict[str, Any],
    detectors: Dict[str, Callable],
    verbose: bool = True
) -> Optional[Callable]:
    """
    Analyze a function and create a quantum version if a pattern is recognized.
    
    Args:
        func: Function to analyze
        components: Dictionary of system components
        detectors: Dictionary mapping pattern names to detector functions
        verbose: Whether to print information
        
    Returns:
        Quantum implementation or None if no pattern detected
    """
    # Use the original function, not a wrapper
    if hasattr(func, '__wrapped__'):
        func = func.__wrapped__
            
    try:
        # Detect quantum patterns
        patterns = analyze_function(func, detectors)
        
        if not patterns:
            return None
        
        # Get the highest confidence pattern
        pattern_name = max(patterns.items(), key=lambda x: x[1])[0]
        confidence = patterns[pattern_name]
        
        if verbose:
            print(f"Detected {pattern_name} pattern in {func.__name__} with confidence {confidence}")
        
        # Create quantum implementation based on the pattern
        return create_quantum_implementation(pattern_name, func, components, verbose)
            
    except Exception as e:
        # If there's an error in analysis, log it but don't crash
        if verbose:
            print(f"Error analyzing function {func.__name__}: {e}")
            import traceback
            traceback.print_exc()
        return None