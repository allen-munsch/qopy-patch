"""
Optimization algorithm pattern detectors.

This module provides detector functions for optimization problems suitable
for quantum optimization algorithms like QAOA and quantum annealing.
"""
import ast
from typing import Callable, Dict, Set, List

# Optimization function names and libraries
OPTIMIZATION_FUNCS = {
    'minimize', 'maximize', 'opt', 'optimize', 'minimise', 'maximise',
    'scipy.optimize.minimize', 'scipy.optimize.basinhopping', 
    'torch.optim', 'optim.Adam', 'optim.SGD'
}

def detect_optimization(tree: ast.AST, func: Callable) -> float:
    """
    Detect optimization algorithm patterns suitable for quantum optimization algorithms.
    
    Args:
        tree: AST of the function
        func: Function object
        
    Returns:
        Confidence score 0-1
    """
    # Check function name for optimization hints
    name_hints = ['optim', 'minimize', 'maximiz', 'cost', 'loss', 'objective']
    name_score = any(hint in func.__name__.lower() for hint in name_hints)
    
    # Look for optimization function calls
    optimization_calls = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check direct function calls
            if isinstance(node.func, ast.Name):
                if any(opt_func in node.func.id.lower() for opt_func in ['minimize', 'maximize', 'optim']):
                    optimization_calls.append(node)
            
            # Check module function calls
            elif isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr') and hasattr(node.func.value, 'id'):
                    call_name = f"{node.func.value.id}.{node.func.attr}"
                    if any(opt_func in call_name.lower() for opt_func in OPTIMIZATION_FUNCS):
                        optimization_calls.append(node)
    
    # Look for gradient calculation patterns
    gradient_patterns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr'):
                    if node.func.attr in ['gradient', 'grad', 'backward', 'differentiate', 'jacobian']:
                        gradient_patterns.append(node)
    
    # Look for iterative improvement patterns (common in optimization)
    has_iterative_improvement = False
    for node in ast.walk(tree):
        if isinstance(node, ast.For) or isinstance(node, ast.While):
            # Look for value updates in the loop
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Assign) or isinstance(subnode, ast.AugAssign):
                    has_iterative_improvement = True
                    break
    
    # Look for objective function definition
    has_objective_function = False
    objective_func_patterns = ['cost', 'loss', 'objective', 'fitness', 'energy', 'error']
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name.lower()
            if any(pattern in func_name for pattern in objective_func_patterns):
                has_objective_function = True
                break
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id.lower()
            if any(pattern in var_name for pattern in objective_func_patterns):
                has_objective_function = True
                break
    
    # Look for combinatorial optimization problems
    combinatorial_patterns = 0
    
    # Check for binary variables (common in quantum optimization)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if isinstance(op, (ast.Eq, ast.In)) and isinstance(node.comparators[i], ast.List):
                    # Check if comparing to 0/1 or True/False
                    list_elements = node.comparators[i].elts
                    if len(list_elements) == 2:
                        if all(isinstance(e, ast.Constant) for e in list_elements):
                            values = [e.value for e in list_elements]
                            if values == [0, 1] or values == [False, True]:
                                combinatorial_patterns += 1
                                
    # Check for constraint expressions
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            # Look for inequality constraints
            for op in node.ops:
                if isinstance(op, (ast.LtE, ast.GtE, ast.Lt, ast.Gt)):
                    combinatorial_patterns += 1
                    break
    
    # Calculate confidence score
    if optimization_calls:
        return 0.9
    elif has_objective_function and has_iterative_improvement:
        return 0.85
    elif gradient_patterns and has_iterative_improvement:
        return 0.8
    elif combinatorial_patterns >= 2:
        return 0.75
    elif has_iterative_improvement and name_score:
        return 0.7
    elif has_iterative_improvement:
        return 0.5
    elif name_score:
        return 0.3
    else:
        return 0