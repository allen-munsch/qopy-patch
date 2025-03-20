# quantum_jit/visualization/pattern_dashboard.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List

def visualize_patterns(pattern_data: List[Dict]):
    """
    Visualize detected patterns across functions.
    
    Args:
        pattern_data: List of dictionaries with keys: 
                      'function', 'pattern', 'confidence', 'speedup'
    """
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
    pivot_df = df.pivot_table(index='function', columns='pattern', values='confidence', fill_value=0)
    sns.heatmap(pivot_df, annot=True, cmap='viridis')
    plt.title('Pattern Detection Confidence by Function')
    
    plt.tight_layout()
    plt.savefig('pattern_detection_dashboard.png')
    plt.show()