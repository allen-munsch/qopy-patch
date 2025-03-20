import numpy as np
import time
from quantum_jit import qjit, visualize_all

# Define several functions with different patterns that can be quantum-accelerated

@qjit
def matrix_multiplication(a, b):
    """Matrix multiplication pattern."""
    return np.dot(a, b)

@qjit
def fourier_transform(x):
    """Fourier transform pattern."""
    return np.fft.fft(x)

@qjit
def search_function(items, target):
    """Search algorithm pattern."""
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1

@qjit
def optimize_binary(objective_func, n_vars):
    """Optimization pattern."""
    best_solution = None
    best_value = float('inf')
    
    # Very simple binary optimization (just for demonstration)
    for i in range(2**n_vars):
        # Convert to binary array
        binary = [(i >> j) & 1 for j in range(n_vars)]
        
        # Evaluate objective function
        value = objective_func(binary)
        
        # Update best solution
        if value < best_value:
            best_value = value
            best_solution = binary
    
    return best_solution, best_value

# Helper function for the optimization example
def sample_objective(x):
    """Simple objective function for binary optimization."""
    # Penalize for too many 1's
    penalty1 = sum(x) * 0.5
    
    # Penalize for adjacent 1's
    penalty2 = sum(x[i] * x[i+1] for i in range(len(x)-1)) * 2
    
    return penalty1 + penalty2

def evaluate_all_functions(runs=3):
    """Run all the quantum-accelerable functions multiple times."""
    
    print("Running matrix multiplication...")
    for _ in range(runs):
        a = np.random.rand(4, 4)
        b = np.random.rand(4, 4)
        result = matrix_multiplication(a, b)
        time.sleep(0.1)  # Add a small delay between calls
    
    print("Running Fourier transform...")
    for _ in range(runs):
        x = np.random.rand(8)
        result = fourier_transform(x)
        time.sleep(0.1)
    
    print("Running search function...")
    for _ in range(runs):
        items = list(range(10, 30))
        target = np.random.choice(items)
        result = search_function(items, target)
        time.sleep(0.1)
    
    print("Running optimization function...")
    for _ in range(runs):
        result = optimize_binary(sample_objective, 6)
        time.sleep(0.1)
    
    print("All functions executed.")

if __name__ == "__main__":
    # Run our quantum-accelerable functions
    evaluate_all_functions(runs=2)
    
    # Generate visualizations
    print("\nGenerating quantum acceleration visualizations...")
    output_dir = "./quantum_analysis"
    visualize_all(output_dir=output_dir)
    
    print(f"\nVisualizations have been saved to: {output_dir}")
    print("The following visualization files were created:")
    print(" - quantum_call_graph.png: Function call graph showing quantum-accelerated functions")
    print(" - pattern_detection_dashboard.png: Distribution of detected patterns")
    print(" - pattern_confidence_vs_speedup.png: Scatterplot of confidence vs. speedup")
    print(" - performance_timeline.png: Timeline of classical vs. quantum performance")
    print(" - performance_summary.png: Summary of performance improvements")
    
    print("\nAnalyzing the visualizations can help you understand:")
    print(" - Which patterns were detected in your code")
    print(" - How much speedup each quantum implementation provides")
    print(" - Which functions benefit most from quantum acceleration")
    print(" - How performance varies over time")
