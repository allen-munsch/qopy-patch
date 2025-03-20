"""
Tests for the pattern detection module.
"""
import unittest
import numpy as np

from quantum_jit.pattern_detection.base_detector import BasePatternDetector
from quantum_jit.pattern_detection.common_patterns import (
    detect_matrix_multiply,
    detect_fourier_transform,
    detect_search
)
from quantum_jit.pattern_detection.function_analyzer import FunctionAnalyzer


class TestPatternDetection(unittest.TestCase):
    """Test cases for pattern detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = BasePatternDetector()
        
        # Register common patterns
        self.detector.register_pattern("matrix_multiplication", detect_matrix_multiply)
        self.detector.register_pattern("fourier_transform", detect_fourier_transform)
        self.detector.register_pattern("search_algorithm", detect_search)
    
    def test_matrix_multiply_detection(self):
        """Test detection of matrix multiplication."""
        def matrix_mult_func(a, b):
            return np.dot(a, b)
        
        def matrix_matmul_func(a, b):
            return a @ b
        
        def not_matrix_func(a, b):
            return a + b
        
        # Test detection
        results1 = self.detector.analyze_function(matrix_mult_func)
        results2 = self.detector.analyze_function(matrix_matmul_func)
        results3 = self.detector.analyze_function(not_matrix_func)
        
        # Check results
        self.assertIn("matrix_multiplication", results1)
        self.assertIn("matrix_multiplication", results2)
        self.assertFalse("matrix_multiplication" in results3)
    
    def test_fourier_transform_detection(self):
        """Test detection of Fourier transform."""
        def fft_func(x):
            return np.fft.fft(x)
        
        def fourier_impl_func(x):
            n = len(x)
            result = np.zeros(n, dtype=complex)
            for k in range(n):
                for j in range(n):
                    result[k] += x[j] * np.exp(-2j * np.pi * k * j / n)
            return result
        
        def not_fourier_func(x):
            return x * 2
        
        # Test detection
        results1 = self.detector.analyze_function(fft_func)
        results2 = self.detector.analyze_function(fourier_impl_func)
        results3 = self.detector.analyze_function(not_fourier_func)
        
        # Check results
        self.assertIn("fourier_transform", results1)
        self.assertIn("fourier_transform", results2)
        self.assertFalse("fourier_transform" in results3)
    
    def test_search_detection(self):
        """Test detection of search algorithms."""
        def search_func(items, target):
            for i, item in enumerate(items):
                if item == target:
                    return i
            return -1
        
        def not_search_func(items):
            return sum(items)
        
        # Test detection
        results1 = self.detector.analyze_function(search_func)
        results2 = self.detector.analyze_function(not_search_func)
        
        # Check results
        self.assertIn("search_algorithm", results1)
        self.assertFalse("search_algorithm" in results2)
    
    def test_analyze_source_code(self):
        """Test analyzing source code directly."""
        source_code = """
def matrix_mult(a, b):
    return np.dot(a, b)

def search_value(array, target):
    for i in range(len(array)):
        if array[i] == target:
            return i
    return -1
        """
        
        # Analyze source code
        results = self.detector.analyze_source(source_code)
        
        # Check results
        self.assertIn("matrix_mult", results)
        self.assertIn("search_value", results)
        self.assertIn("matrix_multiplication", results["matrix_mult"])
        self.assertIn("search_algorithm", results["search_value"])


class TestFunctionAnalyzer(unittest.TestCase):
    """Test cases for function analyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FunctionAnalyzer(num_samples=5)
    
    def test_analyze_linear_function(self):
        """Test analyzing a linear function."""
        def linear_func(x):
            return 2 * x + 1
        
        properties = self.analyzer.analyze_function(linear_func)
        
        self.assertTrue(properties["is_linear"])
        self.assertEqual(properties["output_dimension"], 1)
    
    def test_analyze_nonlinear_function(self):
        """Test analyzing a non-linear function."""
        def nonlinear_func(x):
            return x ** 2
        
        properties = self.analyzer.analyze_function(nonlinear_func)
        
        self.assertFalse(properties["is_linear"])
    
    def test_analyze_symmetric_function(self):
        """Test analyzing a symmetric function."""
        def symmetric_func(x):
            return x ** 2
        
        properties = self.analyzer.analyze_function(symmetric_func)
        
        self.assertTrue(properties["is_symmetric"])
    
    def test_analyze_array_function(self):
        """Test analyzing a function that works with arrays."""
        def array_func(x):
            return np.sum(x)
        
        properties = self.analyzer.analyze_function(
            array_func, input_domain=(-1, 1)
        )
        
        self.assertEqual(properties["output_dimension"], 1)


if __name__ == "__main__":
    unittest.main()