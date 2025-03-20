"""
Cryptography pattern detectors.

This module provides detector functions for cryptographic algorithms
that could benefit from quantum acceleration or are relevant to post-quantum cryptography.
"""
import ast
from typing import Callable, Dict, Set, List

# Cryptography libraries
CRYPTO_LIBRARIES = {
    'cryptography', 'pycrypto', 'pycryptodome', 'Crypto', 'hashlib', 'rsa', 'openssl',
    'hmac', 'secret', 'nacl', 'pyca', 'bcrypt', 'cryptodome', 'M2Crypto',
    'libnacl', 'PyNaCl', 'pysodium', 'pqcrypto'
}

# Cryptographic function names
CRYPTO_FUNCTIONS = {
    'encrypt', 'decrypt', 'sign', 'verify', 'hash', 'sha', 'aes', 'rsa',
    'dsa', 'cipher', 'md5', 'digest', 'hmac', 'pbkdf2', 'blake', 'derive',
    'kdf', 'scrypt', 'bcrypt', 'argon2', 'chacha', 'poly1305', 'gcm'
}

# Post-quantum cryptography algorithms
PQ_ALGORITHMS = {
    'kyber', 'saber', 'ntru', 'mceliece', 'falcon', 'dilithium', 'sphincs',
    'lattice', 'multivariate', 'isogeny', 'supersingular', 'sidh', 'sike'
}

def detect_encryption(tree: ast.AST, func: Callable) -> float:
    """
    Detect encryption/cryptography patterns suitable for quantum cryptography.
    
    Args:
        tree: AST of the function
        func: Function object
        
    Returns:
        Confidence score 0-1
    """
    # Check function name for cryptography hints
    name_hints = ['encrypt', 'decrypt', 'crypt', 'cipher', 'hash', 'sha', 
                 'sign', 'verify', 'secret', 'key', 'secure', 'nonce', 'salt',
                 'hmac', 'signature', 'password', 'digest', 'auth']
    name_score = any(hint in func.__name__.lower() for hint in name_hints)
    
    # Look for cryptography library imports
    crypto_imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in node.names:
                if any(crypto_lib in name.name for crypto_lib in CRYPTO_LIBRARIES):
                    crypto_imports.append(name.name)
    
    # Look for cryptographic function calls
    crypto_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id.lower() in CRYPTO_FUNCTIONS:
                    crypto_calls.append(node)
            elif isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr') and node.func.attr.lower() in CRYPTO_FUNCTIONS:
                    crypto_calls.append(node)
    
    # Check for binary operations (common in crypto)
    bitwise_ops = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitXor, ast.BitOr, ast.BitAnd, ast.LShift, ast.RShift)):
            bitwise_ops.append(node)
    
    # Check for byte/binary data handling
    byte_handling = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ['bytes', 'bytearray']:
                byte_handling.append(node)
            elif isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr') and node.func.attr in ['encode', 'decode', 'hex', 'b64encode', 'b64decode', 'hexdigest']:
                    byte_handling.append(node)
        elif isinstance(node, ast.Bytes):
            byte_handling.append(node)
    
    # Check for post-quantum cryptography
    pq_crypto_elements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and any(pq_algo in node.id.lower() for pq_algo in PQ_ALGORITHMS):
            pq_crypto_elements.append(node)
        elif isinstance(node, ast.Attribute) and hasattr(node, 'attr') and any(pq_algo in node.attr.lower() for pq_algo in PQ_ALGORITHMS):
            pq_crypto_elements.append(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and any(pq_algo in node.value.lower() for pq_algo in PQ_ALGORITHMS):
            pq_crypto_elements.append(node)
    
    # Check for constant-time operations (common in crypto to prevent timing attacks)
    constant_time_patterns = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr') and node.func.attr in ['compare_digest', 'constant_time_compare', 'hmac_compare']:
                    constant_time_patterns = True
                    break
    
    # Check for large number operations (common in asymmetric crypto)
    large_number_ops = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ['pow', 'mod', 'modpow', 'modinv']:
                large_number_ops = True
                break
            elif isinstance(node.func, ast.Attribute):
                if hasattr(node.func, 'attr') and node.func.attr in ['pow', 'mod', 'gcd', 'inverse', 'modpow', 'modinv']:
                    large_number_ops = True
                    break
    
    # Look for key generation or key exchange
    key_operations = False
    key_related_terms = ['key', 'priv', 'pub', 'secret', 'exchange', 'agreement', 'generate', 'derive']
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id.lower()
            if any(term in var_name for term in key_related_terms):
                key_operations = True
                break
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and hasattr(node.func, 'attr'):
                func_name = node.func.attr.lower()
                if 'key' in func_name and any(action in func_name for action in ['gen', 'exchange', 'agree', 'derive']):
                    key_operations = True
                    break
    
    # Check for explicit block/round processing (common in block ciphers)
    block_processing = False
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            # Check for round or block terminology
            if isinstance(node.target, ast.Name):
                if 'round' in node.target.id.lower() or 'block' in node.target.id.lower():
                    block_processing = True
                    break
            
            # Or check for actions within the loop that suggest block processing
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.BinOp) and isinstance(subnode.op, ast.BitXor):
                    block_processing = True
                    break
    
    # Calculate confidence score
    if pq_crypto_elements:
        return 0.95  # Very high confidence for post-quantum crypto
    elif crypto_imports and crypto_calls:
        return 0.9  # High confidence with imports and function calls
    elif crypto_imports and (key_operations or constant_time_patterns):
        return 0.85  # Strong evidence with imports and key operations
    elif crypto_calls and byte_handling:
        return 0.8  # Good evidence with function calls and byte handling
    elif block_processing and bitwise_ops and len(bitwise_ops) > 3:
        return 0.75  # Likely a block cipher implementation
    elif large_number_ops and name_score:
        return 0.7  # Potential asymmetric crypto with naming
    elif key_operations and len(byte_handling) > 2:
        return 0.65  # Key operations with byte handling
    elif crypto_imports and name_score:
        return 0.6  # Crypto imports with crypto-related naming
    elif len(crypto_calls) > 0:
        return 0.55  # Some crypto function calls
    elif len(bitwise_ops) > 3 and len(byte_handling) > 2:
        return 0.5  # Operations common in crypto implementations
    elif constant_time_patterns:
        return 0.45  # Security-conscious code patterns
    elif len(byte_handling) > 2 and name_score:
        return 0.4  # Byte handling with crypto-related naming
    elif crypto_imports:
        return 0.35  # Just crypto imports
    elif large_number_ops and len(bitwise_ops) > 2:
        return 0.3  # Operations that might be crypto-related
    elif name_score:
        return 0.2  # Just crypto-related naming
    else:
        return 0