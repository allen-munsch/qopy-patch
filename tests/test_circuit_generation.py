"""
Tests for the circuit generation module.
"""
import unittest
import numpy as np

from quantum_jit.circuit_generation.circuit_generator import QuantumCircuitGenerator
from quantum_jit.circuit_generation.circuit_optimizer import CircuitOptimizer


class TestCircuitGenerator(unittest.TestCase):
    """Test cases for quantum circuit generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = QuantumCircuitGenerator()
    
    def test_hadamard_circuit_generation(self):
        """Test generating a Hadamard transform circuit."""
        num_qubits = 3
        circuit = self.generator.generate_hadamard_circuit(num_qubits)
        
        # Check circuit properties
        self.assertEqual(circuit.num_qubits, num_qubits)
        self.assertEqual(circuit.num_clbits, num_qubits)
        
        # Check that circuit contains Hadamard gates
        hadamard_count = 0
        for inst, _, _ in circuit.data:
            if inst.name == 'h':
                hadamard_count += 1
        
        self.assertEqual(hadamard_count, num_qubits)
    
    def test_qft_circuit_generation(self):
        """Test generating a QFT circuit."""
        num_qubits = 3
        circuit = self.generator.generate_qft_circuit(num_qubits)
        
        # Check circuit properties
        self.assertEqual(circuit.num_qubits, num_qubits)
        
        # Check that circuit contains Hadamard gates and phase rotations
        hadamard_count = 0
        phase_count = 0
        for inst, _, _ in circuit.data:
            if inst.name == 'h':
                hadamard_count += 1
            elif inst.name == 'cp':
                phase_count += 1
        
        self.assertEqual(hadamard_count, num_qubits)
        self.assertGreater(phase_count, 0)
    
    def test_grover_circuit_generation(self):
        """Test generating a Grover search circuit."""
        num_qubits = 3
        circuit = self.generator.generate_grover_circuit(num_qubits)
        
        # Check circuit properties
        self.assertEqual(circuit.num_qubits, num_qubits)
        self.assertEqual(circuit.num_clbits, num_qubits)
        
        # Check that circuit contains Hadamard gates and multi-controlled operations
        hadamard_count = 0
        for inst, _, _ in circuit.data:
            if inst.name == 'h':
                hadamard_count += 1
        
        # At least 2*num_qubits Hadamard gates (initial + diffusion)
        self.assertGreaterEqual(hadamard_count, 2*num_qubits)
    
    def test_binary_function_circuit_generation(self):
        """Test generating a binary function evaluation circuit."""
        num_qubits = 3
        circuit = self.generator.generate_binary_function_circuit(num_qubits)
        
        # Check circuit properties
        self.assertEqual(circuit.num_qubits, num_qubits + 1)  # Extra qubit for output
        self.assertEqual(circuit.num_clbits, num_qubits)
        
        # Check that circuit contains Hadamard gates and CX gates
        hadamard_count = 0
        cx_count = 0
        for inst, _, _ in circuit.data:
            if inst.name == 'h':
                hadamard_count += 1
            elif inst.name == 'cx':
                cx_count += 1
        
        self.assertGreaterEqual(hadamard_count, num_qubits)
        self.assertGreaterEqual(cx_count, num_qubits)
    
    def test_qaoa_circuit_generation(self):
        """Test generating a QAOA circuit."""
        num_qubits = 3
        problem_hamiltonian = [
            ([0], 1.0),
            ([1], 1.0),
            ([0, 1], 0.5)
        ]
        
        circuit = self.generator.generate_qaoa_circuit(
            problem_hamiltonian=problem_hamiltonian,
            num_qubits=num_qubits,
            p=1
        )
        
        # Check circuit properties
        self.assertEqual(circuit.num_qubits, num_qubits)
        self.assertEqual(circuit.num_clbits, num_qubits)
        
        # Check that circuit contains Hadamard gates and rotations
        hadamard_count = 0
        rx_count = 0
        rz_count = 0
        for inst, _, _ in circuit.data:
            if inst.name == 'h':
                hadamard_count += 1
            elif inst.name == 'rx':
                rx_count += 1
            elif inst.name == 'rz':
                rz_count += 1
        
        self.assertEqual(hadamard_count, num_qubits)  # Initial state preparation
        self.assertEqual(rx_count, num_qubits)  # Mixing Hamiltonian
        self.assertGreaterEqual(rz_count, 2)  # Problem Hamiltonian


class TestCircuitOptimizer(unittest.TestCase):
    """Test cases for quantum circuit optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = CircuitOptimizer()
        self.generator = QuantumCircuitGenerator()
    
    def test_circuit_optimization(self):
        """Test optimizing a quantum circuit."""
        # Create a circuit with some redundancies
        num_qubits = 3
        circuit = self.generator.generate_hadamard_circuit(num_qubits)
        
        # Add some redundant gates (H followed by H cancels out)
        for i in range(num_qubits):
            circuit.h(i)
            circuit.h(i)
        
        # Optimize the circuit
        optimized = self.optimizer.optimize_circuit(circuit)
        
        # The optimized circuit should be simpler
        self.assertLessEqual(optimized.depth(), circuit.depth())
    
    def test_resource_estimation(self):
        """Test estimating resources for a circuit."""
        num_qubits = 3
        circuit = self.generator.generate_hadamard_circuit(num_qubits)
        
        # Get resource estimates
        resources = self.optimizer.estimate_resources(circuit)
        
        # Check that essential metrics are included
        self.assertIn('num_qubits', resources)
        self.assertIn('depth', resources)
        self.assertIn('gate_count', resources)
        
        # Check specific values
        self.assertEqual(resources['num_qubits'], num_qubits)
        self.assertGreater(resources['gate_count'], num_qubits)  # At least H gates + measurements


if __name__ == "__main__":
    unittest.main()