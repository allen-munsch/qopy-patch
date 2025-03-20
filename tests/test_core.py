"""
Tests for the core quantum JIT module.
"""
import unittest
import numpy as np

from quantum_jit.core import QuantumJITCompiler, qjit


class TestQuantumJIT(unittest.TestCase):
    """Test cases for quantum JIT compilation."""

    def setUp(self):
        """Set up test fixtures."""
        self.compiler = QuantumJITCompiler(verbose=False)
    
    def test_jit_decorator(self):
        """Test JIT decorator functionality."""
        # Define a simple function
        @self.compiler.jit
        def add(a, b):
            return a + b
        
        # Call the function
        result = add(2, 3)
        
        # Check result
        self.assertEqual(result, 5)
    
    def test_pattern_detection_and_patching(self):
        """Test detecting patterns and creating quantum implementations."""
        # Define a function with a known pattern (matrix multiplication)
        @self.compiler.jit
        def matrix_multiply(a, b):
            return np.dot(a, b)
        
        # Call once to trigger analysis
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = matrix_multiply(a, b)
        
        # Check if quantum implementation was created
        func_id = id(matrix_multiply.__wrapped__)
        self.assertIn(func_id, self.compiler.quantum_implementations)
    
    def test_performance_tracking(self):
        """Test performance tracking."""
        # Define a function
        @self.compiler.jit
        def hadamard_transform(x):
            n = len(x)
            h_matrix = np.ones((n, n))
            for i in range(n):
                for j in range(n):
                    if bin(i & j).count('1') % 2 == 1:
                        h_matrix[i, j] = -1
            h_matrix = h_matrix / np.sqrt(n)
            return np.dot(h_matrix, x)
        
        # Call once to trigger benchmarking
        x = np.array([1, 0, 0, 0])
        result = hadamard_transform(x)
        
        # Check performance data
        func_id = id(hadamard_transform.__wrapped__)
        self.assertIn(func_id, self.compiler.performance_data)
        
        perf_data = self.compiler.performance_data[func_id]
        self.assertIn('classical_time', perf_data)
        self.assertIn('quantum_time', perf_data)
        self.assertIn('speedup', perf_data)
    
    def test_quantum_implementation_decision(self):
        """Test decision logic for using quantum implementation."""
        # Create a mock performance data entry
        mock_func = lambda x: x
        func_id = id(mock_func)
        
        # First test: speedup below threshold
        self.compiler.performance_data[func_id] = {
            'classical_time': 1.0,
            'quantum_time': 2.0,  # Slower than classical
            'speedup': 0.5,       # Below threshold
            'correct': True
        }
        
        # Should not use quantum
        self.assertFalse(self.compiler._should_use_quantum(func_id, (), {}))
        
        # Second test: speedup above threshold
        self.compiler.performance_data[func_id] = {
            'classical_time': 2.0,
            'quantum_time': 1.0,  # Faster than classical
            'speedup': 2.0,       # Above threshold
            'correct': True
        }
        
        # Mock the quantum implementation
        self.compiler.quantum_implementations[func_id] = mock_func
        
        # Should use quantum
        self.assertTrue(self.compiler._should_use_quantum(func_id, (), {}))
        
        # Third test: incorrect results
        self.compiler.performance_data[func_id] = {
            'classical_time': 2.0,
            'quantum_time': 1.0,  # Faster than classical
            'speedup': 2.0,       # Above threshold
            'correct': False      # But incorrect
        }
        
        # Should not use quantum if results are incorrect
        self.assertFalse(self.compiler._should_use_quantum(func_id, (), {}))
    
    def test_result_comparison(self):
        """Test comparison of classical and quantum results."""
        # Test with numpy arrays
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.01, 1.99, 3.01])  # Close enough
        self.assertTrue(self.compiler._compare_results(arr1, arr2))
        
        arr3 = np.array([1.0, 2.0, 3.0])
        arr4 = np.array([1.5, 2.5, 3.5])  # Too different
        self.assertFalse(self.compiler._compare_results(arr3, arr4))
        
        # Test with dictionaries
        dict1 = {0: 0, 1: 1, 2: 1}
        dict2 = {0: 0, 1: 1, 2: 1}
        self.assertTrue(self.compiler._compare_results(dict1, dict2))
        
        dict3 = {0: 0, 1: 1, 2: 1}
        dict4 = {0: 0, 1: 1, 2: 0}  # Different value for key 2
        self.assertFalse(self.compiler._compare_results(dict3, dict4))
        
        # Test with scalars
        self.assertTrue(self.compiler._compare_results(42, 42))
        self.assertFalse(self.compiler._compare_results(42, 43))


class TestGlobalQJIT(unittest.TestCase):
    """Test cases for the global qjit decorator."""

    def test_global_qjit_decorator(self):
        """Test the global qjit decorator."""
        # Define a function with the global decorator
        @qjit
        def square(x):
            return x * x
        
        # Call the function
        result = square(5)
        
        # Check result
        self.assertEqual(result, 25)
    
    def test_qjit_with_parameters(self):
        """Test qjit decorator with parameters."""
        # Define a function with parameterized decorator
        @qjit(auto_patch=False, verbose=False)
        def cube(x):
            return x * x * x
        
        # Call the function
        result = cube(3)
        
        # Check result
        self.assertEqual(result, 27)


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for quantum JIT."""

    def test_hadamard_transform(self):
        """Test end-to-end Hadamard transform."""
        @qjit(verbose=False)
        def hadamard_transform(input_vector):
            n = len(input_vector)
            h_matrix = np.ones((n, n))
            for i in range(n):
                for j in range(n):
                    if bin(i & j).count('1') % 2 == 1:
                        h_matrix[i, j] = -1
            h_matrix = h_matrix / np.sqrt(n)
            return np.dot(h_matrix, input_vector)
        
        # Input vector
        input_vector = np.zeros(8)
        input_vector[0] = 1
        
        # First call to benchmark
        result1 = hadamard_transform(input_vector)
        
        # Expected result for Hadamard transform of |0⟩
        expected = np.ones(8) / np.sqrt(8)
        
        # Check result
        self.assertTrue(np.allclose(result1, expected, rtol=1e-6))
        
        # Second call may use quantum
        result2 = hadamard_transform(input_vector)
        
        # Should still be correct
        self.assertTrue(np.allclose(result2, expected, rtol=1e-6))
    
    def test_binary_function_evaluation(self):
        """Test end-to-end binary function evaluation."""
        @qjit(verbose=False)
        def evaluate_all(f, n):
            results = {}
            for x in range(2**n):
                results[x] = f(x)
            return results
        
        # Parity function
        def parity(x):
            return bin(x).count('1') % 2
        
        # First call to benchmark
        result1 = evaluate_all(parity, 3)
        
        # Expected result
        expected = {x: parity(x) for x in range(8)}
        
        # Check result
        self.assertEqual(result1, expected)
        
        # Second call may use quantum
        result2 = evaluate_all(parity, 3)
        
        # Should still be correct
        self.assertEqual(result2, expected)
    
    def test_search_algorithm(self):
        """Test end-to-end search algorithm."""
        @qjit(verbose=False)
        def find_item(items, target):
            for i, item in enumerate(items):
                if item == target:
                    return i
            return -1
        
        # Items to search
        items = [10, 20, 30, 40, 50, 60, 70, 80]
        target = 50
        
        # First call to benchmark
        result1 = find_item(items, target)
        
        # Expected result
        expected = 4  # Index of 50
        
        # Check result
        self.assertEqual(result1, expected)
        
        # Second call may use quantum
        result2 = find_item(items, target)
        
        # Should still be correct
        self.assertEqual(result2, expected)
        
        # Test not finding an item
        result3 = find_item(items, 100)
        self.assertEqual(result3, -1)


if __name__ == "__main__":
    unittest.main()