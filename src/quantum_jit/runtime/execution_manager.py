"""
Manages execution of quantum circuits with batching and prioritization.
"""
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from queue import PriorityQueue


class CircuitQueueItem:
    """Class to hold circuit items in the priority queue with proper comparison."""
    
    def __init__(self, priority: int, data: Dict[str, Any]):
        self.priority = priority
        self.data = data
    
    def __lt__(self, other):
        # Compare based on priority
        return self.priority < other.priority
    
    def __eq__(self, other):
        # Compare based on priority
        return self.priority == other.priority

class ExecutionManager:
    """
    Manages execution of quantum circuits with optimizations.
    
    Features:
    - Batches similar circuits to reduce overhead
    - Prioritizes circuits based on estimated importance
    - Manages backend connections and job submission
    """
    
    def __init__(self, backend_name: str = 'qasm_simulator', max_batch_size: int = 5):
        """
        Initialize the execution manager.
        
        Args:
            backend_name: Name of the quantum backend to use
            max_batch_size: Maximum number of circuits in a batch
        """
        self.backend = Aer.get_backend(backend_name)
        self.max_batch_size = max_batch_size
        self.pending_circuits = PriorityQueue()
        self.results_cache = {}

    def execute_circuit(self, circuit: QuantumCircuit, shots: int = 1024,
                    priority: int = 1) -> str:
        """
        Submit a circuit for execution.
        
        Args:
            circuit: Quantum circuit to execute
            shots: Number of shots
            priority: Priority level (lower = higher priority)
            
        Returns:
            Job ID for retrieving results
        """
        # Generate a unique job ID
        job_id = f"job_{time.time()}_{id(circuit)}"
        
        # Create data dictionary
        circuit_data = {
            'circuit': circuit,
            'shots': shots,
            'job_id': job_id
        }
        
        # Queue the circuit for execution using CircuitQueueItem for proper comparison
        self.pending_circuits.put(CircuitQueueItem(
            priority,
            circuit_data
        ))
        
        # Process batch if enough circuits or high priority
        if self.pending_circuits.qsize() >= self.max_batch_size or priority == 0:
            self._process_batch()
        
        return job_id

    def get_result(self, job_id: str, blocking: bool = True,
                 timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get the result of a circuit execution.
        
        Args:
            job_id: Job ID
            blocking: Whether to wait for the result
            timeout: Maximum time to wait in seconds
            
        Returns:
            Result dictionary or None if not ready
        """
        start_time = time.time()
        
        while blocking and job_id not in self.results_cache:
            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                return None
                
            # Process any pending batches
            if not self.pending_circuits.empty():
                self._process_batch()
                
            # Small delay to avoid CPU spinning
            time.sleep(0.1)
        
        # Return result if available
        return self.results_cache.get(job_id)

    def _process_batch(self) -> None:
        """Process a batch of pending circuits."""
        # Extract circuits to process
        batch = []
        batch_ids = []
        
        while not self.pending_circuits.empty() and len(batch) < self.max_batch_size:
            # Get the item from the queue - it's now a CircuitQueueItem, not a tuple
            queue_item = self.pending_circuits.get()
            item = queue_item.data
            batch.append(item['circuit'])
            batch_ids.append(item['job_id'])
        
        if not batch:
            return
        
        # Group similar circuits
        # (In a more sophisticated implementation, we would analyze circuits
        # and group those with similar structure together)
        
        # Execute the batch
        transpiled_circuits = transpile(batch, self.backend)
        job = self.backend.run(transpiled_circuits)
        results = job.result()
        
        # Store results
        for i, job_id in enumerate(batch_ids):
            counts = results.get_counts(i)
            self.results_cache[job_id] = {
                'counts': counts,
                'time': time.time()
            }

    def clear_old_results(self, max_age_seconds: float = 3600) -> None:
        """
        Clear results older than a specified age.
        
        Args:
            max_age_seconds: Maximum age in seconds
        """
        current_time = time.time()
        keys_to_remove = []
        
        for job_id, result in self.results_cache.items():
            if current_time - result['time'] > max_age_seconds:
                keys_to_remove.append(job_id)
        
        for job_id in keys_to_remove:
            del self.results_cache[job_id]
