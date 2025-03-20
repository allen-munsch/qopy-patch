# quantum_jit/visualization/code_annotator.py
import ast
import tokenize
import io
from typing import Dict, Set

def generate_annotations(source_code: str, 
                        quantum_accelerated_funcs: Set[str],
                        pattern_data: Dict[str, Dict]) -> str:
    """
    Generate annotations for source code to highlight quantum acceleration.
    
    Args:
        source_code: Python source code
        quantum_accelerated_funcs: Set of function names that are accelerated
        pattern_data: Pattern detection data by function name
        
    Returns:
        HTML-formatted code with annotations
    """
    tree = ast.parse(source_code)
    annotations = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in quantum_accelerated_funcs:
            pattern = pattern_data.get(node.name, {}).get('pattern', 'Unknown')
            speedup = pattern_data.get(node.name, {}).get('speedup', 0)
            
            annotations.append({
                'line': node.lineno,
                'text': f"Quantum Accelerated: {pattern} pattern (Speedup: {speedup:.2f}x)"
            })
    
    # Generate HTML with annotations
    html_lines = []
    for i, line in enumerate(source_code.splitlines(), 1):
        annotation = next((a for a in annotations if a['line'] == i), None)
        
        if annotation:
            html_lines.append(f'<div class="line quantum-accelerated" data-tooltip="{annotation["text"]}">{line}</div>')
        else:
            html_lines.append(f'<div class="line">{line}</div>')
    
    return "\n".join(html_lines)