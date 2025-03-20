"""
General algorithm pattern detectors.

This module provides detector functions for general algorithms
that could benefit from quantum acceleration, including sorting
and sampling algorithms.
"""
import ast
from typing import Callable, Dict, Set, List

# Sorting algorithm names and related terms
SORTING_TERMS = {
    'sort', 'quicksort', 'mergesort', 'heapsort', 'bubblesort', 'insertionsort',
    'selectionsort', 'radixsort', 'shellsort', 'bucketsort', 'countingsort',
    'order', 'arrange', 'rank'
}

# Sampling algorithm names and related terms
SAMPLING_TERMS = {
    'sample', 'random', 'monte', 'carlo', 'mcmc', 'markov', 'metropolis',
    'gibbs', 'bootstrap', 'resample', 'permutation', 'draw', 'distribution'
}

def detect_sorting(tree: ast.AST, func: Callable) -> float:
    """
    Detect sorting algorithm patterns.
    
    Args:
        tree: AST of the function
        func: Function object
        
    Returns:
        Confidence score 0-1
    """
    # Check function name for sorting hints
    name_score = any(term in func.__name__.lower() for term in SORTING_TERMS)
    
    # Look for built-in sorting function calls
    sorting_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'sorted':
                sorting_calls.append(node)
            elif isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr') and node.func.attr == 'sort':
                    sorting_calls.append(node)
    
    # Look for comparison patterns (common in sorting algorithms)
    comparisons = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            # Check for comparisons between adjacent elements or with a pivot
            if isinstance(node.left, ast.Subscript) and any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops):
                comparisons.append(node)
    
    # Check for nested loops (common in many O(n²) sorting algorithms)
    nested_loops = 0
    class LoopVisitor(ast.NodeVisitor):
        def __init__(self):
            self.loop_depth = 0
            self.max_loop_depth = 0
            self.nested_loops = 0
            
        def visit_For(self, node):
            self.loop_depth += 1
            self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
            if self.loop_depth > 1:
                self.nested_loops += 1
            self.generic_visit(node)
            self.loop_depth -= 1
            
        def visit_While(self, node):
            self.loop_depth += 1
            self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
            if self.loop_depth > 1:
                self.nested_loops += 1
            self.generic_visit(node)
            self.loop_depth -= 1
    
    loop_visitor = LoopVisitor()
    loop_visitor.visit(tree)
    nested_loops = loop_visitor.nested_loops
    max_loop_depth = loop_visitor.max_loop_depth
    
    # Check for array swapping operations (common in sorting algorithms)
    swap_operations = 0
    for node in ast.walk(tree):
        # Look for temporary variable swaps
        if isinstance(node, ast.Assign):
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id.lower() in ['temp', 'tmp', 'swap', 't']:
                swap_operations += 1
        
        # Look for direct subscript assignments that might be swaps
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            # Check if this is part of a sequence of subscript assignments (potential swap)
            if isinstance(node.value, ast.Subscript):
                swap_operations += 1
    
    # Check for divide-and-conquer patterns (common in merge sort, quicksort)
    has_recursion = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == func.__name__:
                has_recursion = True
                break
    
    # Check for partition or merge operations
    has_partition_or_merge = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if any(term in node.name.lower() for term in ['partition', 'merge', 'pivot']):
                has_partition_or_merge = True
                break
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if any(term in node.targets[0].id.lower() for term in ['pivot', 'partition', 'merged']):
                has_partition_or_merge = True
                break
    
    # Calculate confidence score
    if (sorting_calls and name_score) or (name_score and has_partition_or_merge):
        return 0.9  # High confidence with explicit sorting calls/functions and name
    elif has_recursion and has_partition_or_merge:
        return 0.85  # Likely recursive divide-and-conquer sorting algorithm
    elif nested_loops >= 1 and swap_operations >= 2 and len(comparisons) >= 2:
        return 0.8  # Strong evidence of an iterative sorting algorithm
    elif len(comparisons) >= 3 and swap_operations >= 1 and name_score:
        return 0.7  # Good evidence with comparisons, swaps, and naming
    elif sorting_calls:
        return 0.6  # Moderate evidence with just sorting function calls
    elif has_recursion and len(comparisons) >= 2:
        return 0.5  # Some evidence of recursive algorithm with comparisons
    elif nested_loops >= 1 and len(comparisons) >= 2:
        return 0.4  # Weak evidence of iterative comparison-based algorithm
    elif swap_operations >= 2 and name_score:
        return 0.3  # Minimal evidence with swaps and sorting-related naming
    elif name_score:
        return 0.2  # Very weak evidence based on name only
    else:
        return 0

def detect_sampling(tree: ast.AST, func: Callable) -> float:
    """
    Detect sampling algorithm patterns suitable for quantum sampling.
    
    Args:
        tree: AST of the function
        func: Function object
        
    Returns:
        Confidence score 0-1
    """
    # Check function name for sampling hints
    name_score = any(term in func.__name__.lower() for term in SAMPLING_TERMS)
    
    # Look for random number generation
    random_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr') and node.func.attr in ['random', 'choice', 'sample', 'randint', 'uniform', 'randn', 'normal', 'exponential']:
                    if hasattr(node.func.value, 'id') and node.func.value.id in ['random', 'np', 'numpy', 'rng', 'rand']:
                        random_calls.append(node)
            elif isinstance(node.func, ast.Name) and node.func.id in ['random', 'choice', 'sample', 'randint']:
                random_calls.append(node)
    
    # Look for probability distributions
    distribution_patterns = []
    distribution_names = ['normal', 'gaussian', 'uniform', 'exponential', 'poisson', 'binomial', 
                         'beta', 'gamma', 'multinomial', 'dirichlet', 'bernoulli']
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr') and any(dist in node.func.attr.lower() for dist in distribution_names):
                    distribution_patterns.append(node)
            elif isinstance(node, ast.Name) and any(dist in node.id.lower() for dist in distribution_names):
                distribution_patterns.append(node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if any(dist in node.targets[0].id.lower() for dist in distribution_names):
                distribution_patterns.append(node)
    
    # Look for Monte Carlo patterns
    monte_carlo_patterns = False
    # Check for repeated random sampling with accumulation
    has_accumulation = False
    has_loops_with_random = False
    
    for node in ast.walk(tree):
        # Check for loops containing random calls
        if isinstance(node, (ast.For, ast.While)):
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Call):
                    if isinstance(subnode.func, ast.Attribute) and hasattr(subnode.func, 'attr'):
                        if subnode.func.attr in ['random', 'choice', 'sample', 'randint']:
                            has_loops_with_random = True
                            break
        
        # Check for accumulation of results
        if isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Add, ast.Sub)):
            has_accumulation = True
    
    if has_loops_with_random and has_accumulation:
        monte_carlo_patterns = True
    
    # Check for MCMC-specific patterns
    has_mcmc_patterns = False
    mcmc_terms = ['acceptance', 'proposal', 'metropolis', 'gibbs', 'hastings', 'markov', 'chain', 'transition']
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id.lower()
            if any(term in var_name for term in mcmc_terms):
                has_mcmc_patterns = True
                break
    
    # Look for acceptance/rejection patterns
    has_accept_reject = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Check condition for comparison with random value (common in acceptance/rejection)
            if isinstance(node.test, ast.Compare):
                for subnode in ast.walk(node.test):
                    if isinstance(subnode, ast.Call):
                        if isinstance(subnode.func, ast.Attribute) and hasattr(subnode.func, 'attr'):
                            if subnode.func.attr in ['random', 'uniform']:
                                has_accept_reject = True
                                break
    
    # Calculate confidence score
    if has_mcmc_patterns and random_calls and monte_carlo_patterns:
        return 0.95  # Very high confidence for MCMC with random sampling in loops
    elif monte_carlo_patterns and name_score:
        return 0.9  # High confidence for Monte Carlo with appropriate naming
    elif len(random_calls) >= 3 and has_loops_with_random:
        return 0.85  # Strong evidence of repeated random sampling
    elif has_accept_reject and random_calls:
        return 0.8  # Good evidence of acceptance/rejection sampling
    elif len(distribution_patterns) >= 2 and random_calls:
        return 0.75  # Sampling from multiple distributions
    elif monte_carlo_patterns:
        return 0.7  # Evidence of Monte Carlo without explicit naming
    elif len(random_calls) >= 2 and name_score:
        return 0.6  # Multiple random calls with sampling-related naming
    elif has_mcmc_patterns or (has_accept_reject and has_accumulation):
        return 0.5  # Some MCMC or acceptance/rejection patterns
    elif random_calls and has_loops_with_random:
        return 0.4  # Random generation in loops
    elif len(random_calls) >= 1:
        return 0.3  # Some random generation
    elif name_score:
        return 0.2  # Just sampling-related naming
    else:
        return 0