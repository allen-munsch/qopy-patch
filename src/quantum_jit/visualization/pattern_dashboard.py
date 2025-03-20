"""
Pattern detection dashboard visualization.
"""
from typing import Dict, List, Any
import os


def visualize_patterns(pattern_data: List[Dict],
                      output_dir='./quantum_viz'):
    """
    Visualize detected patterns across functions.
    
    Args:
        pattern_data: List of dictionaries with keys: 
                      'function', 'pattern', 'confidence', 'speedup', 'timestamp'
        output_dir: Directory to save visualization files
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(pattern_data)
        
        # Create pattern distribution chart
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        pattern_counts = df['pattern'].value_counts()
        sns.barplot(x=pattern_counts.index, y=pattern_counts.values)
        plt.title('Distribution of Detected Patterns')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Create confidence heatmap
        plt.subplot(2, 1, 2)
        if len(df) > 1:  # Only create pivot if we have enough data
            pivot_df = df.pivot_table(index='function', columns='pattern', values='confidence', fill_value=0)
            sns.heatmap(pivot_df, annot=True, cmap='viridis')
            plt.title('Pattern Detection Confidence by Function')
        else:
            # Just display the confidence values
            plt.bar(df['function'], df['confidence'])
            plt.title('Pattern Detection Confidence')
            plt.ylabel('Confidence')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pattern_detection_dashboard.png'), dpi=300)
        plt.close()
        
        # Create a second figure for speedup vs. confidence
        plt.figure(figsize=(10, 6))
        g = sns.scatterplot(data=df, x='confidence', y='speedup', hue='pattern', 
                           size='speedup', sizes=(50, 200), alpha=0.7)
        
        for i, row in df.iterrows():
            plt.text(row['confidence'] + 0.01, row['speedup'], row['function'], 
                    fontsize=9, alpha=0.8)
        
        plt.title('Pattern Confidence vs. Quantum Speedup')
        plt.xlabel('Pattern Detection Confidence')
        plt.ylabel('Speedup Factor')
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break-even point')
        plt.legend(title='Pattern')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pattern_confidence_vs_speedup.png'), dpi=300)
        plt.close()
        
    except ImportError as e:
        print(f"Pattern dashboard visualization requires additional dependencies: {e}")
        print("Please install: pip install matplotlib pandas seaborn")