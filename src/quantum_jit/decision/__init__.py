"""
Decision making for quantum implementation selection.
"""
from quantum_jit.decision.decision_maker import (
    should_use_quantum,
    compare_results
)

__all__ = [
    'should_use_quantum',
    'compare_results'
]