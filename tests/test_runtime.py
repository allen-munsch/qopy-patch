"""
Tests for the runtime module.
"""
import unittest
import time
import numpy as np
from qiskit import QuantumCircuit

from quantum_jit.runtime.circuit_cache import CircuitCache
from quantum_jit.runtime.execution_manager import ExecutionManager
from quantum_jit.runtime.result_processor import ResultProcessor


class TestCircuitCache(unittest.TestCase):
    """Test cases for circuit caching."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = CircuitCache(max_size=5)
    
    def test_cache_storage_and_retrieval(self):
        """Test storing and retrieving circuits from cache."""
        # Create a simple circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        # Store in cache
        func_id = 12345
        input_shape = (2, 2)
        self.cache.store_circuit(func_id, input_shape, qc)
        
        # Retrieve from cache
        retrieved = self.cache.get_circuit(func_id, input_shape)
        
        # Check that we got the same circuit back
        self.assertEqual(retrieved.num_qubits, qc.num_qubits)
        self.assertEqual(len(retrieved.data), len(qc.data))
    
    def test_cache_eviction(self):
        """Test cache eviction when full."""
        # Create a cache with tiny capacity
        small_cache = CircuitCache(max_size=2)
        
        # Create circuits
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        
        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.cx(0, 1)
        
        qc3 = QuantumCircuit(3)
        qc3.h(0)
        qc3.cx(0, 1)
        qc3.cx(1, 2)
        
        # Store circuits
        small_cache.store_circuit(1, (1,), qc1)
        small_cache.store_circuit(2, (2,), qc2)
        
        # Check both are cached
        self.assertIsNotNone(small_cache.get_circuit(1, (1,)))
        self.assertIsNotNone(small_cache.get_circuit(2, (2,)))
        
        # Store one more (should evict the oldest)
        small_cache.store_circuit(3, (3,), qc3)
        
        # Check what's in cache now
        self.assertIsNone(small_cache.get_circuit(1, (1,)))  # Should be evicted
        self.assertIsNotNone(small_cache.get_circuit(2, (2,)))
        self.assertIsNotNone(small_cache.get_circuit(3, (3,)))
    
    def test_cache_statistics(self):
        """Test cache hit/miss statistics."""
        # Create a circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        # Store in cache
        self.cache.store_circuit(1, (2,), qc)
        
        # Access a few times
        self.cache.get_circuit(1, (2,))  # Hit
        self.cache.get_circuit(1, (2,))  # Hit
        self.cache.get_circuit(2, (2,))  # Miss
        
        # Check statistics
        stats = self.cache.get_stats()
        self.assertEqual(stats['hits'], 2)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['size'], 1)


class TestExecutionManager(unittest.TestCase):
    """Test cases for execution manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = ExecutionManager()
    
    def test_circuit_execution(self):
        """Test executing a quantum circuit."""
        # Create a simple circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        
        # Execute the circuit
        job_id = self.manager.execute_circuit(qc)
        
        # Get results (blocking)
        result = self.manager.get_result(job_id)
        
        # Check results
        self.assertIsNotNone(result)
        self.assertIn('counts', result)
        self.assertGreater(len(result['counts']), 0)
    
    def test_circuit_batch_execution(self):
        """Test executing a batch of circuits."""
        # Create several simple circuits
        circuits = []
        for i in range(3):
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            if i > 0:
                qc.x(1)  # Make them slightly different
            qc.measure([0, 1], [0, 1])
            circuits.append(qc)
        
        # Execute all circuits
        job_ids = [self.manager.execute_circuit(qc) for qc in circuits]
        
        # Get all results
        results = [self.manager.get_result(job_id) for job_id in job_ids]
        
        # Check all results
        for result in results:
            self.assertIsNotNone(result)
            self.assertIn('counts', result)
    
    def test_circuit_execution_timeout(self):
        """Test execution timeout."""
        # Create a simple circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        
        # Execute the circuit with high priority to ensure it processes
        job_id = self.manager.execute_circuit(qc, priority=0)
        
        # Get results with a very short timeout
        result = self.manager.get_result(job_id, timeout=5.0)
        
        # Should still get a result since the circuit is simple
        self.assertIsNotNone(result)


class TestResultProcessor(unittest.TestCase):
    """Test cases for result processor."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = ResultProcessor()
    
    def test_hadamard_results_processing(self):
        """Test processing Hadamard transform results."""
        # Simulate measurement outcomes from Hadamard transform
        counts = {
            '000': 128,
            '001': 128,
            '010': 128,
            '011': 128,
            '100': 128,
            '101': 128,
            '110': 128,
            '111': 128
        }
        
        # Process results
        def original_hadamard(x):
            return np.ones(8) / np.sqrt(8)
        
        result = self.processor.process_results(
            counts, 'hadamard', original_hadamard, shots=1024
        )
        
        # Check that result is a vector with uniform values
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 8)
        self.assertTrue(np.allclose(result, np.ones(8) / np.sqrt(8), rtol=0.1))
    
    def test_binary_function_results_processing(self):
        """Test processing binary function evaluation results."""
        # Simulate measurement outcomes
        counts = {
            '000': 128,
            '001': 128,
            '010': 128,
            '011': 128,
            '100': 128,
            '101': 128,
            '110': 128,
            '111': 128
        }
        
        # Define original function (parity)
        def parity(x):
            return bin(x).count('1') % 2
        
        # Process results
        result = self.processor.process_results(
            counts, 'binary_function', parity, shots=1024
        )
        
        # Check results
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 8)
        for i in range(8):
            self.assertEqual(result[i], parity(i))
    
    def test_function_expects_integer(self):
        """Test detection of function input types."""
        # Function that expects integer
        def func_int(x):
            return x * 2
        
        # Function that expects list/array
        def func_list(x):
            return sum(x)
        
        # Check detection
        self.assertTrue(self.processor._function_expects_integer(func_int))
        self.assertFalse(self.processor._function_expects_integer(func_list))


if __name__ == "__main__":
    unittest.main()