"""
Graph algorithm pattern detectors.

This module provides detector functions for graph algorithms
that could benefit from quantum acceleration.
"""
import ast
from typing import Callable, Dict, Set, List

# Graph libraries and frameworks
GRAPH_LIBRARIES = {
    'networkx', 'igraph', 'graph_tool', 'pygraph', 'pydot', 'graphviz',
    'snap', 'networkit', 'jgrapht', 'networkit', 'pynauty', 'digraph',
    'gensim', 'stellargraph', 'dgl', 'torch_geometric', 'pyg', 'spektral',
    'ogb', 'graphnet'
}

# Graph algorithm function names
GRAPH_FUNCTIONS = {
    'shortest_path', 'connected_components', 'pagerank', 'clique', 
    'community', 'centrality', 'coloring', 'matching', 'flow',
    'dijkstra', 'bellman_ford', 'floyd_warshall', 'kruskal', 'prim',
    'max_flow', 'min_cut', 'betweenness', 'closeness', 'degree'
}

def detect_graph_algorithm(tree: ast.AST, func: Callable) -> float:
    """
    Detect graph algorithm patterns suitable for quantum graph algorithms.
    
    Args:
        tree: AST of the function
        func: Function object
        
    Returns:
        Confidence score 0-1
    """
    # Check function name for graph algorithm hints
    name_hints = ['graph', 'network', 'path', 'connect', 'clique', 'color', 
                 'community', 'centrality', 'degree', 'adjacency', 'vertex',
                 'node', 'edge', 'neighbor', 'distance', 'dijkstra']
    name_score = any(hint in func.__name__.lower() for hint in name_hints)
    
    # Look for graph library imports
    graph_imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in node.names:
                if any(graph_lib in name.name for graph_lib in GRAPH_LIBRARIES):
                    graph_imports.append(name.name)
    
    # Look for graph algorithm function calls
    graph_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr'):
                    if any(graph_func in node.func.attr for graph_func in GRAPH_FUNCTIONS):
                        graph_calls.append(node)
    
    # Look for adjacency matrix / adjacency list patterns
    adjacency_patterns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id.lower()
                if any(pattern in var_name for pattern in ['adj', 'edge', 'neighbor', 'graph']):
                    adjacency_patterns.append(node)
        
        # Look for dictionary or list of lists (common graph representations)
        elif isinstance(node, ast.Dict) or (isinstance(node, ast.List) and 
                                           node.elts and isinstance(node.elts[0], ast.List)):
            for parent in ast.iter_child_nodes(tree):
                if isinstance(parent, ast.Assign) and node in ast.iter_child_nodes(parent):
                    if isinstance(parent.targets[0], ast.Name):
                        var_name = parent.targets[0].id.lower()
                        if any(pattern in var_name for pattern in ['adj', 'edge', 'neighbor', 'graph']):
                            adjacency_patterns.append(node)
    
    # Check for nested loops with array/dictionary access (common in graph traversal)
    class LoopVisitor(ast.NodeVisitor):
        def __init__(self):
            self.loop_depth = 0
            self.max_loop_depth = 0
            self.nested_loops = 0
            self.has_subscript_in_loop = False
            
        def visit_For(self, node):
            self.loop_depth += 1
            self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
            if self.loop_depth > 1:
                self.nested_loops += 1
            
            # Check for subscript operations inside the loop (accessing graph elements)
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Subscript):
                    self.has_subscript_in_loop = True
            
            self.generic_visit(node)
            self.loop_depth -= 1
            
        def visit_While(self, node):
            self.loop_depth += 1
            self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
            if self.loop_depth > 1:
                self.nested_loops += 1
            
            # Check for subscript operations inside the loop
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Subscript):
                    self.has_subscript_in_loop = True
            
            self.generic_visit(node)
            self.loop_depth -= 1
    
    loop_visitor = LoopVisitor()
    loop_visitor.visit(tree)
    nested_loops = loop_visitor.nested_loops
    max_loop_depth = loop_visitor.max_loop_depth
    has_subscript_in_loop = loop_visitor.has_subscript_in_loop
    
    # Check for queue/stack usage (common in graph traversal)
    has_queue_or_stack = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id.lower()
                if any(pattern in var_name for pattern in ['queue', 'stack', 'frontier', 'visited']):
                    has_queue_or_stack = True
                    break
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr') and node.func.attr in ['append', 'pop', 'popleft', 'push']:
                    has_queue_or_stack = True
                    break
    
    # Check for graph creation patterns
    has_graph_creation = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ['Graph', 'DiGraph', 'MultiGraph']:
                has_graph_creation = True
                break
            elif isinstance(node.func, ast.Attribute) and hasattr(node.func, 'attr'):
                if node.func.attr in ['Graph', 'DiGraph', 'MultiGraph']:
                    has_graph_creation = True
                    break
    
    # Check for vertex/edge iteration patterns
    has_vertex_edge_iteration = False
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            if isinstance(node.iter, ast.Call):
                if isinstance(node.iter.func, ast.Attribute):
                    if hasattr(node.iter.func, 'attr') and node.iter.func.attr in ['nodes', 'edges', 'neighbors', 'successors', 'predecessors']:
                        has_vertex_edge_iteration = True
                        break
    
    # Calculate confidence score
    if graph_imports and graph_calls:
        return 0.95  # Very high confidence with imports and function calls
    elif has_graph_creation and has_vertex_edge_iteration:
        return 0.9  # High confidence with graph creation and traversal
    elif graph_imports and adjacency_patterns:
        return 0.85  # High confidence with imports and adjacency patterns
    elif adjacency_patterns and nested_loops > 0 and has_subscript_in_loop:
        return 0.8  # Strong evidence of graph traversal
    elif adjacency_patterns and has_queue_or_stack:
        return 0.75  # Likely graph BFS/DFS implementation
    elif max_loop_depth >= 2 and has_subscript_in_loop and name_score:
        return 0.7  # Good evidence with nested loops and graph-related naming
    elif has_graph_creation or graph_calls:
        return 0.65  # Moderate evidence with graph creation or functions
    elif adjacency_patterns and name_score:
        return 0.6  # Some evidence with adjacency patterns and naming
    elif graph_imports:
        return 0.5  # Weak evidence with just imports
    elif max_loop_depth >= 2 and name_score:
        return 0.4  # Minimal evidence with nested loops and naming
    elif has_queue_or_stack and name_score:
        return 0.3  # Very weak evidence
    elif name_score:
        return 0.2  # Name-only evidence
    else:
        return 0