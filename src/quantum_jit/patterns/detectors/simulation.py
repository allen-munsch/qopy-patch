"""
Quantum simulation pattern detectors.

This module provides detector functions for patterns related to quantum system
simulation and physical simulations that could benefit from quantum acceleration.
"""
import ast
from typing import Callable, Dict, Set, List

# Physics and quantum simulation terms
PHYSICS_SIMULATION_TERMS = {
    'hamiltonian', 'eigen', 'schrodinger', 'quantum', 'fermion', 'boson',
    'spin', 'lattice', 'unitary', 'hermitian', 'evolution', 'eigenvalue',
    'eigenstate', 'ground_state', 'density_matrix', 'pauli'
}

# Physics and quantum simulation libraries
PHYSICS_MODULES = {
    'qutip', 'qiskit', 'cirq', 'pyquil', 'pennylane', 
    'scipy.sparse', 'scipy.linalg', 'quspin', 'openfermion',
    'netket', 'pyscf', 'tenpy', 'quimb'
}

def detect_quantum_simulation(tree: ast.AST, func: Callable) -> float:
    """
    Detect patterns related to quantum system simulation.
    
    Args:
        tree: AST of the function
        func: Function object
        
    Returns:
        Confidence score 0-1
    """
    # Check function name for physics/quantum hints
    name_score = any(term in func.__name__.lower() for term in PHYSICS_SIMULATION_TERMS)
    
    # Look for physics/quantum related imports
    has_physics_imports = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            for name in node.names:
                module_name = name.name
                if any(physics_mod in module_name for physics_mod in PHYSICS_MODULES):
                    has_physics_imports = True
                    break
    
    # Look for matrix operations (common in quantum simulation)
    matrix_ops = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr'):
                    if node.func.attr in ['dot', 'matmul', 'multiply', 'diag', 'eig', 'eigvals', 
                                         'expm', 'eigh', 'kron', 'tensordot']:
                        matrix_ops.append(node)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            matrix_ops.append(node)
    
    # Look for physics terms in variable names and string literals
    physics_terms_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id.lower()
            if any(term in var_name for term in PHYSICS_SIMULATION_TERMS):
                physics_terms_count += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(term in node.value.lower() for term in PHYSICS_SIMULATION_TERMS):
                physics_terms_count += 1
    
    # Check for matrix creation patterns (common in Hamiltonian construction)
    matrix_creation = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr'):
                    if node.func.attr in ['array', 'zeros', 'ones', 'eye', 'identity', 'matrix', 'diag']:
                        if hasattr(node.func.value, 'id'):
                            if node.func.value.id in ['np', 'numpy', 'scipy', 'sp']:
                                matrix_creation = True
            elif isinstance(node.func, ast.Name):
                if node.func.id in ['array', 'zeros', 'ones', 'eye', 'identity', 'matrix', 'diag']:
                    matrix_creation = True
    
    # Check for exponential operations (common in time evolution)
    has_exponential = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr') and hasattr(node.func.value, 'id'):
                    if node.func.attr == 'exp' and node.func.value.id in ['np', 'numpy', 'math']:
                        has_exponential = True
                    elif node.func.attr == 'expm' and node.func.value.id in ['scipy', 'sp', 'linalg']:
                        has_exponential = True
    
    # Check for complex number usage
    has_complex_numbers = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, complex):
            has_complex_numbers = True
            break
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'complex':
            has_complex_numbers = True
            break
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and isinstance(node.right, ast.Name):
            if hasattr(node.right, 'id') and node.right.id == 'j':
                has_complex_numbers = True
                break
    
    # Look for tensor product patterns
    has_tensor_product = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr'):
                    if node.func.attr in ['kron', 'tensordot', 'tensor', 'outer']:
                        has_tensor_product = True
                        break
    
    # Look for common quantum simulation terms in comments
    comments_with_physics_terms = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            comment = node.value.value.lower()
            for term in PHYSICS_SIMULATION_TERMS:
                if term in comment:
                    comments_with_physics_terms += 1
                    break
    
    # Calculate confidence score
    if has_physics_imports and (physics_terms_count > 1 or has_exponential):
        return 0.95  # Very high confidence for direct quantum simulation imports with related terms
    elif has_physics_imports:
        return 0.9  # High confidence just for imports
    elif physics_terms_count >= 3:
        return 0.85  # High confidence for multiple physics terms
    elif matrix_ops and has_complex_numbers and has_exponential:
        return 0.8  # Strong evidence of quantum or physics simulation
    elif has_tensor_product and has_complex_numbers:
        return 0.75  # Likely quantum simulation
    elif matrix_creation and has_complex_numbers and name_score:
        return 0.7  # Possible quantum simulation
    elif matrix_ops and physics_terms_count > 0:
        return 0.6
    elif has_complex_numbers and physics_terms_count > 0:
        return 0.5
    elif matrix_ops and name_score:
        return 0.4
    elif physics_terms_count > 0:
        return 0.3
    elif name_score or comments_with_physics_terms > 0:
        return 0.2
    else:
        return 0