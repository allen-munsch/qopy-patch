#!/usr/bin/env python
"""
Example demonstrating how to use the quantum JIT system with visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_jit import qjit

def run_jit_with_visualization():
    # Define a function with Hadamard transform pattern
    @qjit(verbose=True)
    def hadamard_transform(input_vector):
        n = len(input_vector)
        h_matrix = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if bin(i & j).count('1') % 2 == 1:
                    h_matrix[i, j] = -1
        h_matrix = h_matrix / np.sqrt(n)
        return np.dot(h_matrix, input_vector)
    
    # Input vector - |0⟩ state in 3-qubit system
    input_vector = np.zeros(8)
    input_vector[0] = 1
    
    print("Running Hadamard transform using JIT system...")
    
    # First call to trigger analysis and benchmarking
    result1 = hadamard_transform(input_vector)
    
    # Print the result
    print("\nHadamard transform of |000⟩:")
    print(result1)
    
    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(result1)), np.abs(result1))
    plt.xticks(range(len(result1)), [f"{i:03b}" for i in range(len(result1))])
    plt.xlabel("State")
    plt.ylabel("Amplitude")
    plt.title("Hadamard Transform of |000⟩")
    plt.savefig('jit_hadamard_result.png')
    plt.close()
    
    # Try another input
    input_vector2 = np.zeros(8)
    input_vector2[1] = 1  # |001⟩ state
    
    # Second call may use quantum implementation if beneficial
    result2 = hadamard_transform(input_vector2)
    
    # Print the result
    print("\nHadamard transform of |001⟩:")
    print(result2)
    
    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(result2)), np.abs(result2))
    plt.xticks(range(len(result2)), [f"{i:03b}" for i in range(len(result2))])
    plt.xlabel("State")
    plt.ylabel("Amplitude")
    plt.title("Hadamard Transform of |001⟩")
    plt.savefig('jit_hadamard_result2.png')
    plt.close()
    
    # Define a function with search pattern
    @qjit(verbose=True)
    def search_array(items, target):
        for i, item in enumerate(items):
            if item == target:
                return i
        return -1
    
    # Create array to search
    search_items = np.array([3, 7, 2, 9, 1, 5, 6, 4])
    target = 5
    
    print("\nRunning search using JIT system...")
    
    # First call to trigger analysis and benchmarking
    search_result = search_array(search_items, target)
    
    print(f"Searching for {target} in {search_items}")
    print(f"Found at index: {search_result}")

if __name__ == "__main__":
    run_jit_with_visualization()
