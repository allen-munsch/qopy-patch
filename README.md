# Qopy Patch

Quantum Copy-and-Patch JIT

A Python library for detecting classical algorithms that could benefit from quantum acceleration and dynamically replacing them with quantum implementations. Just does pattern detection via ASTs currently uses QisKit, but could use other libraries with effort.

## Overview

Quantum Copy-and-Patch JIT analyzes your Python functions at runtime to identify common patterns that are suitable for quantum acceleration, such as matrix multiplication, Fourier transforms, search algorithms, and more. When beneficial, it can automatically replace classical implementations with quantum versions.

Key features:
- Automatic detection of quantum-accelerable patterns in your code
- Performance benchmarking to ensure quantum implementations are only used when they provide a speedup
- Circuit caching to reduce overhead for repeated function calls
- Visualization tools to understand which functions are being accelerated

## Installation

```bash
pip install qopy-patch
```


## Basic Usage

Simply add the `@qjit` decorator to functions you want to potentially accelerate:

```python
import numpy as np
from quantum_jit import qjit

@qjit
def matrix_multiply(a, b):
    return np.dot(a, b)

# First call benchmarks classical vs quantum
result = matrix_multiply(np.random.rand(4, 4), np.random.rand(4, 4))

# Subsequent calls may use quantum if faster
result = matrix_multiply(np.random.rand(4, 4), np.random.rand(4, 4))
```

## Supported Patterns

The library can currently detect and accelerate these algorithm patterns:
- Matrix multiplication
- Fourier transforms
- Search algorithms (similar to Grover's algorithm)
- Binary optimization problems
- General linear algebra operations

## Visualization

To visualize which functions are being quantum-accelerated:

```python
from quantum_jit import visualize_all

# Run your quantum-accelerated functions...

# Then generate visualizations
visualize_all(output_dir="./quantum_viz")
```

This generates graphs showing:
- Which functions have been quantum-accelerated
- The confidence of pattern detection
- Performance comparisons between classical and quantum implementations
- Speedup trends across multiple function calls

## Limitations

This library has some important limitations to be aware of:

- **Experimental**: This is research software and not intended for production use.
- **Performance**: Current quantum implementations may not actually be faster than classical ones on available hardware/simulators.
- **Limited Pattern Detection**: Only recognizes specific algorithm patterns.
- **Quantum Simulation**: Runs on quantum simulators by default, not actual quantum hardware.
- **Hardware Requirements**: Visualizations require additional dependencies (matplotlib, pandas, etc.).

## Example

```python
import numpy as np
from quantum_jit import qjit, visualize_all

@qjit
def search_function(items, target):
    """Search algorithm pattern."""
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1

# Run the function a few times
items = list(range(10, 30))
target = 15
result = search_function(items, target)

# Generate visualizations
visualize_all()
```

## License

MIT License

## Requirements

- Python 3.13+
- NumPy
- Qiskit
- Qiskit Aer (for simulation)

For visualization:
- Matplotlib
- Pandas
- Seaborn
- NetworkX

