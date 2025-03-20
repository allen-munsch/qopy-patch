# quantum_jit/visualization/call_graph.py
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Set, Callable

def create_call_graph(quantum_implementations: Dict[int, Callable],
                     performance_data: Dict[int, Dict],
                     function_registry: Dict[int, str]):
    """Create a visualization of function call relationships with quantum acceleration."""
    G = nx.DiGraph()
    
    # Add nodes (functions)
    for func_id, func_name in function_registry.items():
        is_quantum = func_id in quantum_implementations
        speedup = performance_data.get(func_id, {}).get('speedup', 0)
        
        # Add more attributes to nodes for visualization
        G.add_node(func_name, 
                  is_quantum=is_quantum, 
                  speedup=speedup)
    
    # Render the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Color nodes based on whether they're quantum-accelerated
    node_colors = ['lightblue' if G.nodes[n]['is_quantum'] else 'lightgray' 
                  for n in G.nodes]
    
    # Size nodes based on speedup
    node_sizes = [300 + 100 * G.nodes[n].get('speedup', 0) for n in G.nodes]
    
    nx.draw_networkx(G, pos, node_color=node_colors, node_size=node_sizes)
    
    plt.title("Function Call Graph with Quantum Acceleration")
    plt.savefig('quantum_acceleration_graph.png')
    plt.show()