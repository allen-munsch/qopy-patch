# quantum_jit/visualization/performance_tracker.py
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List

def visualize_performance_timeline(performance_history: List[Dict]):
    """
    Visualize performance improvements over time.
    
    Args:
        performance_history: List of dictionaries with keys:
                            'timestamp', 'function', 'classical_time', 
                            'quantum_time', 'speedup'
    """
    df = pd.DataFrame(performance_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set up the figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot execution times
    df.pivot(index='timestamp', columns='function', values='classical_time').plot(
        ax=axes[0], marker='o', linestyle='--', alpha=0.7)
    df.pivot(index='timestamp', columns='function', values='quantum_time').plot(
        ax=axes[0], marker='x', alpha=0.7)
    
    axes[0].set_title('Execution Time Comparison')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].legend(title='Function (solid=quantum, dashed=classical)')
    
    # Plot speedup
    df.pivot(index='timestamp', columns='function', values='speedup').plot(
        ax=axes[1], marker='o')
    
    axes[1].set_title('Quantum Speedup')
    axes[1].set_ylabel('Speedup Factor')
    axes[1].set_xlabel('Time')
    axes[1].axhline(y=1.0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('performance_timeline.png')
    plt.show()