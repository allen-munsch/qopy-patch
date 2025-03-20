"""
Machine learning pattern detectors.

This module provides detector functions for machine learning algorithms
that could benefit from quantum acceleration.
"""
import ast
from typing import Callable, Dict, Set, List

# Machine learning libraries
ML_LIBRARIES = {
    'sklearn', 'tensorflow', 'torch', 'keras', 'xgboost', 'lightgbm',
    'pandas', 'numpy', 'scipy', 'statsmodels', 'theano', 'jax', 'mxnet',
    'dask.ml', 'skorch', 'fastai', 'catboost', 'pyro', 'transformers'
}

# Machine learning function names
ML_FUNCTIONS = {
    'fit', 'predict', 'transform', 'train', 'evaluate', 'classify',
    'cluster', 'regression', 'optimize', 'forward', 'backward'
}

# Common ML model types
ML_MODEL_TYPES = {
    'SVM', 'RandomForest', 'DecisionTree', 'Linear', 'Logistic', 
    'NeuralNetwork', 'BayesianNetwork', 'CNN', 'RNN', 'LSTM', 
    'Transformer', 'KMeans', 'GaussianMixture', 'PCA', 'SVD',
    'GradientBoosting', 'XGBoost', 'LightGBM'
}

def detect_machine_learning(tree: ast.AST, func: Callable) -> float:
    """
    Detect machine learning patterns suitable for quantum ML.
    
    Args:
        tree: AST of the function
        func: Function object
        
    Returns:
        Confidence score 0-1
    """
    # Check function name for ML hints
    name_hints = ['model', 'train', 'predict', 'classify', 'regress', 'cluster', 
                 'neural', 'learn', 'ml', 'ai', 'inference', 'embedding']
    name_score = any(hint in func.__name__.lower() for hint in name_hints)
    
    # Look for ML library imports
    ml_imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in node.names:
                if any(ml_lib in name.name for ml_lib in ML_LIBRARIES):
                    ml_imports.append(name.name)
    
    # Look for ML-related function calls
    ml_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr'):
                    if node.func.attr in ML_FUNCTIONS:
                        ml_calls.append(node.func.attr)
    
    # Look for model creation patterns
    model_creation = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if any(model_class in node.func.id for model_class in ML_MODEL_TYPES):
                    model_creation = True
                    break
            elif isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr'):
                    if any(model_class in node.func.attr for model_class in ML_MODEL_TYPES):
                        model_creation = True
                        break
    
    # Look for dataset manipulation patterns
    data_manipulation = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr'):
                    data_ops = ['split', 'reshape', 'transpose', 'normalize', 'standardize', 
                               'one_hot', 'get_dummies', 'preprocess']
                    if node.func.attr in data_ops:
                        data_manipulation.append(node)
    
    # Look for loss function definitions
    has_loss_function = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            loss_names = ['loss', 'cost', 'error', 'objective']
            if any(name in node.name.lower() for name in loss_names):
                has_loss_function = True
                break
        elif isinstance(node, ast.Assign):
            if isinstance(node.targets[0], ast.Name):
                loss_names = ['loss', 'cost', 'error', 'criterion']
                if any(name in node.targets[0].id.lower() for name in loss_names):
                    has_loss_function = True
                    break
    
    # Look for batching patterns (common in ML)
    has_batching = False
    batch_names = ['batch', 'mini_batch', 'dataloader']
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and any(name in node.id.lower() for name in batch_names):
            has_batching = True
            break
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if any(name in node.targets[0].id.lower() for name in batch_names):
                has_batching = True
                break
    
    # Look for epoch-based training loops
    has_epochs = False
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                if 'epoch' in node.target.id.lower():
                    has_epochs = True
                    break
            # Check for range() with a variable that might be epochs
            elif isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == 'range' and len(node.iter.args) > 0:
                    if isinstance(node.iter.args[0], ast.Name) and 'epoch' in node.iter.args[0].id.lower():
                        has_epochs = True
                        break
    
    # Check for common tensor operations
    tensor_ops = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr'):
                    ops = ['matmul', 'dot', 'mm', 'bmm', 'conv', 'linear', 'softmax', 
                         'relu', 'sigmoid', 'tanh', 'dropout']
                    if node.func.attr in ops:
                        tensor_ops.append(node)
    
    # Calculate confidence score
    if ml_imports and model_creation and ml_calls:
        return 0.95  # Very high confidence with imports, model creation, and ML function calls
    elif ml_imports and ml_calls:
        return 0.9  # High confidence with imports and ML function calls
    elif model_creation and has_loss_function and has_epochs:
        return 0.85  # High confidence for model creation with training loop
    elif ml_imports and tensor_ops and (has_batching or has_epochs):
        return 0.8  # Strong evidence of ML
    elif tensor_ops and has_loss_function:
        return 0.75  # Likely ML code
    elif model_creation or (len(ml_calls) >= 2):
        return 0.7  # Good evidence of ML
    elif ml_imports and name_score:
        return 0.6  # Moderate evidence with imports and naming
    elif len(data_manipulation) >= 2 and name_score:
        return 0.5  # Some evidence with data prep and naming
    elif ml_imports:
        return 0.4  # Weak evidence with just imports
    elif len(ml_calls) > 0:
        return 0.3  # Minimal evidence with some ML function calls
    elif name_score:
        return 0.2  # Very weak evidence based on name only
    else:
        return 0