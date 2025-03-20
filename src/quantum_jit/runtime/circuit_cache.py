"""
Cache for quantum circuits to avoid regeneration.
"""
import time
from typing import Dict, Any, Tuple, Optional
from qiskit import QuantumCircuit

class CircuitCache:
    """Cache for quantum circuits to avoid regeneration."""
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the circuit cache.
        
        Args:
            max_size: Maximum number of circuits to store
        """
        self.circuits = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get_circuit(self, func_id: int, input_shape: Tuple,
                  params: Optional[Dict] = None) -> Optional[QuantumCircuit]:
        """
        Get cached circuit or return None if not available.
        
        Args:
            func_id: Function ID
            input_shape: Shape of input data
            params: Optional parameters
            
        Returns:
            Cached circuit or None
        """
        key = self._make_key(func_id, input_shape, params)
        if key in self.circuits:
            self.hits += 1
            self.circuits[key]['last_used'] = time.time()
            self.circuits[key]['use_count'] += 1
            return self.circuits[key]['circuit']
        
        self.misses += 1
        return None
    
    def store_circuit(self, func_id: int, input_shape: Tuple,
                     circuit: QuantumCircuit, params: Optional[Dict] = None) -> None:
        """
        Store circuit in cache.
        
        Args:
            func_id: Function ID
            input_shape: Shape of input data
            circuit: Quantum circuit to store
            params: Optional parameters
        """
        key = self._make_key(func_id, input_shape, params)
        
        # Check if we need to evict
        if len(self.circuits) >= self.max_size:
            self._evict_circuit()
        
        self.circuits[key] = {
            'circuit': circuit,
            'last_used': time.time(),
            'use_count': 0
        }
    
    def _make_key(self, func_id: int, input_shape: Tuple,
                 params: Optional[Dict] = None) -> str:
        """
        Create a unique key for the function and input.
        
        Args:
            func_id: Function ID
            input_shape: Shape of input data
            params: Optional parameters
            
        Returns:
            Unique key string
        """
        return f"{func_id}_{input_shape}_{hash(str(params)) if params else 'none'}"
    
    def _evict_circuit(self) -> None:
        """Remove least used circuit from cache."""
        # Find the least recently used circuit
        lru_key = min(self.circuits.keys(), 
                      key=lambda k: self.circuits[k]['last_used'])
        
        # Remove it
        del self.circuits[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'size': len(self.circuits),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.circuits.clear()
        # Keep hit/miss stats
