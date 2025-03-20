"""
Processes and adapts results from quantum circuit execution.
"""
import numpy as np
from typing import Dict, Any, List, Union, Callable, Optional
import inspect

class ResultProcessor:
    """Process and adapt results from quantum execution."""
    
    def __init__(self):
        """Initialize the result processor."""
        # Register processors for different types of quantum algorithms
        self.processors = {
            'hadamard': self.process_hadamard_results,
            'qft': self.process_qft_results,
            'grover': self.process_grover_results,
            'binary_function': self.process_binary_function_results,
            'optimization': self.process_optimization_results
        }
    
    def process_results(self, counts: Dict[str, int], algorithm_type: str,
                       original_func: Callable, shots: int = 1024,
                       params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Process circuit execution results.
        
        Args:
            counts: Measurement counts from circuit execution
            algorithm_type: Type of algorithm used
            original_func: Original classical function
            shots: Number of shots used
            params: Additional parameters
            
        Returns:
            Processed result matching original function's output type
        """
        if algorithm_type not in self.processors:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        
        # Process the results based on algorithm type
        processed = self.processors[algorithm_type](counts, original_func, shots, params)
        
        # Try to adapt to the expected output type of the original function
        return self._adapt_result_type(processed, original_func)
        
    def process_hadamard_results(self, counts: Dict[str, int],
                                original_func: Callable, shots: int,
                                params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Process Hadamard transform results.
        
        Args:
            counts: Measurement counts
            original_func: Original classical function
            shots: Number of shots
            params: Additional parameters
            
        Returns:
            Transformed vector
        """
        # Get dimension of the output
        # Fix: convert keys view to list or use next() to get the first key
        n = len(next(iter(counts.keys()))) if counts else 0
        dim = 2**n
        
        # Initialize output vector
        output = np.zeros(dim)
        
        # Convert counts to probabilities
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            output[idx] = count / shots
        
        # Scale appropriately
        output = output * np.sqrt(dim)
        
        return output

    def process_qft_results(self, counts: Dict[str, int],
                        original_func: Callable, shots: int,
                        params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Process Quantum Fourier Transform results.
        
        Args:
            counts: Measurement counts
            original_func: Original classical function
            shots: Number of shots
            params: Additional parameters
            
        Returns:
            Transformed vector
        """
        # Similar to Hadamard but with complex values
        # Fix: Use next(iter(counts.keys())) instead of counts.keys()[0]
        n = len(next(iter(counts.keys()))) if counts else 0
        dim = 2**n
        
        # Initialize output vector
        output = np.zeros(dim, dtype=complex)
        
        # Convert counts to probabilities and phases
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            output[idx] = complex(count / shots, 0)
        
        return output

    def process_grover_results(self, counts: Dict[str, int],
                            original_func: Callable, shots: int,
                            params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Process Grover search results.
        
        Args:
            counts: Measurement counts
            original_func: Original classical function
            shots: Number of shots
            params: Additional parameters
            
        Returns:
            Found item
        """
        # Get the most frequent measurement outcome
        max_bitstring = max(counts, key=counts.get) if counts else None
        
        # Convert to integer
        if max_bitstring:
            result = int(max_bitstring, 2)
            
            # If we have items list in params, return the corresponding item
            items = params.get('items', []) if params else []
            
            # Handle NumPy arrays safely
            if isinstance(items, np.ndarray):
                if items.size > 0 and result < items.size:
                    return items[result]
            elif items and result < len(items):
                return items[result]
                
            return result
        
        return None

    def process_binary_function_results(self, counts: Dict[str, int],
                                    original_func: Callable, shots: int,
                                    params: Optional[Dict[str, Any]] = None) -> Dict[int, Any]:
        """
        Process binary function evaluation results.
        
        Args:
            counts: Measurement counts
            original_func: Original classical function
            shots: Number of shots
            params: Additional parameters
            
        Returns:
            Dictionary mapping inputs to function outputs
        """
        # Determine if function expects an integer or a binary list
        expects_integer = self._function_expects_integer(original_func)
        
        # Number of bits
        # Fix: Use next(iter(counts.keys())) instead of counts.keys()[0]
        n = len(next(iter(counts.keys()))) if counts else 0
        
        # Process each measurement outcome
        results = {}
        for bitstring, count in counts.items():
            x = int(bitstring, 2)
            
            if expects_integer:
                results[x] = original_func(x)
            else:
                # Convert to binary array
                binary_x = [(x >> j) & 1 for j in range(n)]
                results[x] = original_func(binary_x)
        
        return results

    def process_optimization_results(self, counts: Dict[str, int],
                                original_func: Callable, shots: int,
                                params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Process optimization results.
        
        Args:
            counts: Measurement counts
            original_func: Original classical function
            shots: Number of shots
            params: Additional parameters
            
        Returns:
            Optimal solution
        """
        # Extract objective function and number of variables
        obj_func = params.get('objective_func', None) if params else None
        num_vars = params.get('num_vars', None) if params else None
        
        if obj_func is None or num_vars is None:
            # Can't process without these parameters
            return original_func(*params.get('args', []), **params.get('kwargs', {}))
        
        # Number of bits
        n = len(next(iter(counts.keys()))) if counts else 0
        
        # Find best solution
        best_solution = None
        best_value = float('inf')
        
        for bitstring, count in counts.items():
            x = int(bitstring, 2)
            
            # Convert to binary array
            binary_x = [(x >> j) & 1 for j in range(n)]
            
            # Evaluate objective function
            value = obj_func(binary_x) if obj_func else 0
            
            # Update best if improved
            if value < best_value:
                best_value = value
                best_solution = binary_x
        
        # Match original function output format
        try:
            # Create a sample input to check the return type
            # This was the issue - optimize_binary needs two args
            original_result = None
            if 'args' in params:
                original_result = original_func(*params['args'])
            else:
                # Assuming obj_func and num_vars are the required arguments
                original_result = original_func(obj_func, num_vars)
                
            if isinstance(original_result, tuple):
                return best_solution, best_value
            else:
                return best_solution
        except Exception:
            # If we can't determine, return as tuple
            return best_solution, best_value

    def _function_expects_integer(self, func: Callable) -> bool:
        """Determine if a function expects an integer or a list/array."""
        try:
            # Try with a simple integer
            func(1)
            return True
        except (TypeError, ValueError):
            return False
        except Exception:
            # If any other exception, assume it's not due to type mismatch
            return True
    
    def _adapt_result_type(self, result: Any, original_func: Callable) -> Any:
        """
        Adapt result to match the expected output type of the original function.
        
        Args:
            result: Processed result
            original_func: Original classical function
            
        Returns:
            Adapted result
        """
        # Try to infer return type of original function
        try:
            # Create a simple input that should work
            if hasattr(original_func, '__code__'):
                arg_count = original_func.__code__.co_argcount
                sample_input = [1] * arg_count
                original_result = original_func(*sample_input)
                
                # Convert result to match original type
                if isinstance(original_result, np.ndarray) and not isinstance(result, np.ndarray):
                    return np.array(result)
                elif isinstance(original_result, list) and not isinstance(result, list):
                    return list(result)
                
        except Exception:
            # If we can't determine, return as is
            pass
        
        return result

