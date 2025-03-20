# quantum_jit/runtime/backends.py
"""
Backend providers for quantum circuit execution.
"""
from typing import Dict, Any, Optional, Union
from qiskit import QuantumCircuit

class BackendProvider:
    """Base class for quantum backend providers."""
    
    def __init__(self, **kwargs):
        """Initialize the backend provider."""
        self.options = kwargs
        
    def execute_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, Any]:
        """Execute a quantum circuit on this backend."""
        raise NotImplementedError("Backend providers must implement execute_circuit")
        
    def get_backend_name(self) -> str:
        """Get the name of this backend."""
        raise NotImplementedError("Backend providers must implement get_backend_name")


class AerBackendProvider(BackendProvider):
    """Qiskit Aer simulator backend provider."""
    
    def __init__(self, backend_name: str = 'qasm_simulator', **kwargs):
        """Initialize the Aer backend provider."""
        super().__init__(**kwargs)
        from qiskit_aer import Aer
        self.backend = Aer.get_backend(backend_name)
        self.backend_name = backend_name
        
    def execute_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, Any]:
        """Execute a circuit on the Aer simulator."""
        from qiskit import transpile
        
        transpiled = transpile(circuit, self.backend)
        job = self.backend.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        
        return {
            'counts': counts,
            'backend': self.backend_name,
            'shots': shots
        }
        
    def get_backend_name(self) -> str:
        """Get the name of this backend."""
        return self.backend_name


class IBMQuantumBackendProvider(BackendProvider):
    """IBM Quantum backend provider for real quantum hardware."""
    
    def __init__(self, backend_name: str, token: Optional[str] = None, 
                channel: str = 'ibm_quantum', **kwargs):
        """
        Initialize the IBM Quantum backend provider.
        
        Args:
            backend_name: Name of the IBM Quantum backend to use
            token: IBM Quantum API token (if None, will look in environment variables)
            channel: IBM Quantum channel ('ibm_quantum' or 'ibm_cloud')
            **kwargs: Additional options for the backend
        """
        super().__init__(**kwargs)
        
        # Import here to make it optional
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        # Connect to IBM Quantum
        self.service = QiskitRuntimeService(channel=channel, token=token)
        self.backend = self.service.backend(backend_name)
        self.backend_name = backend_name
        
    def execute_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, Any]:
        """Execute a circuit on IBM Quantum hardware."""
        from qiskit import transpile
        
        # Transpile for the specific backend
        transpiled = transpile(circuit, self.backend)
        
        # Run on the backend
        job = self.backend.run(transpiled, shots=shots)
        
        # Wait for the job to complete (this may take a while)
        result = job.result()
        counts = result.get_counts(circuit)
        
        return {
            'counts': counts,
            'backend': self.backend_name,
            'shots': shots,
            'job_id': job.job_id()
        }
        
    def get_backend_name(self) -> str:
        """Get the name of this backend."""
        return self.backend_name


# Factory function to create the appropriate backend
def create_backend(provider: str, **kwargs) -> BackendProvider:
    """
    Create a backend provider based on the specified provider name.
    
    Args:
        provider: 'aer' or 'ibm_quantum'
        **kwargs: Backend-specific configuration options
        
    Returns:
        An initialized BackendProvider instance
    """
    if provider.lower() == 'aer':
        return AerBackendProvider(**kwargs)
    elif provider.lower() == 'ibm_quantum':
        return IBMQuantumBackendProvider(**kwargs)
    else:
        raise ValueError(f"Unknown backend provider: {provider}")