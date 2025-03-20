"""
Performance tracking visualization for quantum acceleration.
"""
from typing import Dict, List, Any
import os


def visualize_performance_timeline(performance_history: List[Dict],
                                 output_dir='./quantum_viz'):
    """
    Visualize performance improvements over time.
    
    Args:
        performance_history: List of dictionaries with keys:
                            'timestamp', 'function', 'classical_time', 
                            'quantum_time', 'speedup'
        output_dir: Directory to save visualization files
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        from matplotlib.ticker import MaxNLocator
        import numpy as np
        
        # Convert to DataFrame
        df = pd.DataFrame(performance_history)
        
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Set up the figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Get unique functions
        functions = df['function'].unique()
        
        # Color map for consistent colors across plots
        colors = plt.cm.tab10(np.linspace(0, 1, len(functions)))
        color_map = dict(zip(functions, colors))
        
        # Plot execution times (scatter plot with lines)
        for i, func in enumerate(functions):
            func_df = df[df['function'] == func]
            
            # Plot classical times
            axes[0].plot(func_df['timestamp'], func_df['classical_time'], 
                       marker='o', linestyle='--', alpha=0.7, color=color_map[func],
                       label=f"{func} (Classical)")
            
            # Plot quantum times
            axes[0].plot(func_df['timestamp'], func_df['quantum_time'], 
                       marker='x', alpha=0.7, color=color_map[func],
                       label=f"{func} (Quantum)")
        
        axes[0].set_title('Execution Time Comparison')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_yscale('log')  # Log scale for better visibility
        axes[0].legend(title='Function', loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot speedup
        for func in functions:
            func_df = df[df['function'] == func]
            axes[1].plot(func_df['timestamp'], func_df['speedup'], 
                       marker='o', color=color_map[func], label=func)
            
            # Add annotations with function names
            for _, row in func_df.iterrows():
                axes[1].annotate(f"{row['speedup']:.1f}x", 
                              (row['timestamp'], row['speedup']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8)
        
        axes[1].set_title('Quantum Speedup')
        axes[1].set_ylabel('Speedup Factor')
        axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break-even')
        axes[1].legend(title='Function', loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        # Ensure integer y-tick values for speedup
        axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_timeline.png'), dpi=300)
        plt.close()
        
        # Create a summary figure
        plt.figure(figsize=(10, 6))
        summary_df = df.groupby('function').agg({
            'classical_time': 'mean',
            'quantum_time': 'mean',
            'speedup': 'max'
        }).reset_index()
        
        # Sort by speedup
        summary_df = summary_df.sort_values('speedup', ascending=False)
        
        # Plot bars for classical and quantum times
        bar_width = 0.35
        index = range(len(summary_df))
        
        plt.bar([i - bar_width/2 for i in index], summary_df['classical_time'], 
              width=bar_width, label='Classical', color='lightgray')
        plt.bar([i + bar_width/2 for i in index], summary_df['quantum_time'], 
              width=bar_width, label='Quantum', color='lightblue')
        
        # Add speedup annotations
        for i, row in enumerate(summary_df.itertuples()):
            plt.text(i, max(row.classical_time, row.quantum_time) + 0.05, 
                  f"{row.speedup:.1f}x", ha='center', va='bottom',
                  fontweight='bold', color='green' if row.speedup > 1 else 'red')
        
        plt.xticks(index, summary_df['function'], rotation=45, ha='right')
        plt.ylabel('Average Execution Time (seconds)')
        plt.title('Quantum vs. Classical Performance Summary')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=300)
        plt.close()
        
    except ImportError as e:
        print(f"Performance visualization requires additional dependencies: {e}")
        print("Please install: pip install matplotlib pandas numpy")