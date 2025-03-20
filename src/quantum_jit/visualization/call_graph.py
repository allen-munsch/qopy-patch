"""
Call graph visualization for quantum acceleration.
"""
from typing import Dict, Any, Callable
import os

def create_call_graph(quantum_implementations: Dict[int, Callable],
                     performance_data: Dict[int, Dict],
                     function_registry: Dict[int, str],
                     output_dir='./quantum_viz'):
    """
    Create a visualization of function call relationships with quantum acceleration.
    
    Args:
        quantum_implementations: Dictionary mapping function IDs to quantum implementations
        performance_data: Dictionary mapping function IDs to performance data
        function_registry: Dictionary mapping function IDs to function names
        output_dir: Directory to save visualization files
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes (functions)
        for func_id, func_name in function_registry.items():
            is_quantum = func_id in quantum_implementations
            perf_data = performance_data.get(func_id, {})
            speedup = perf_data.get('speedup', 0)
            
            # Add more attributes to nodes for visualization
            G.add_node(func_name, 
                      is_quantum=is_quantum, 
                      speedup=speedup)
        
        # Render the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Color nodes based on whether they're quantum-accelerated
        quantum_nodes = [n for n in G.nodes if G.nodes[n].get('is_quantum', False)]
        classical_nodes = [n for n in G.nodes if not G.nodes[n].get('is_quantum', False)]
        
        # Size nodes based on speedup
        node_sizes = {n: 300 + 100 * G.nodes[n].get('speedup', 0) for n in G.nodes}
        
        # Draw quantum nodes
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=quantum_nodes,
                              node_color='lightblue',
                              node_size=[node_sizes[n] for n in quantum_nodes])
        
        # Draw classical nodes
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=classical_nodes,
                              node_color='lightgray',
                              node_size=[node_sizes[n] for n in classical_nodes])
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos)
        
        # Add legend
        speedup_legend = plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='lightblue', markersize=10,
                                  label='Quantum Accelerated')
        classical_legend = plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='lightgray', markersize=10,
                                    label='Classical')
        plt.legend(handles=[speedup_legend, classical_legend], loc='upper right')
        
        plt.title("Function Call Graph with Quantum Acceleration")
        plt.axis('off')
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'quantum_call_graph.png'), dpi=300)
        plt.close()
        
    except ImportError as e:
        print(f"Call graph visualization requires additional dependencies: {e}")
        print("Please install: pip install networkx matplotlib")