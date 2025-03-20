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
