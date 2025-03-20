"""
Optimize quantum circuits for improved performance.
"""
from qiskit import QuantumCircuit, transpile
import numpy as np
from typing import Dict, Optional, List, Union

class CircuitOptimizer:
    """Optimize quantum circuits for better performance."""
    
    def __init__(self, optimization_level: int = 3):
        """
        Initialize the circuit optimizer.
        
        Args:
            optimization_level: Level of optimization (0-3)
        """
        self.optimization_level = optimization_level
    
    def optimize_circuit(self, circuit: QuantumCircuit,
                        target_basis: Optional[List[str]] = None) -> QuantumCircuit:
        """
        Optimize a quantum circuit.
        
        Args:
            circuit: Quantum circuit to optimize
            target_basis: Optional target gate set
            
        Returns:
            Optimized quantum circuit
        """
        # Use Qiskit's transpiler for basic optimization
        optimized = transpile(
            circuit,
            basis_gates=target_basis,
            optimization_level=self.optimization_level
        )
        
        # Apply custom optimizations
        optimized = self._custom_optimizations(optimized)
        
        return optimized
    
    def _custom_optimizations(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply custom circuit optimizations."""
        # Apply gate cancellation
        circuit = self._cancel_adjacent_gates(circuit)
        
        # Apply common subcircuit recognition
        circuit = self._recognize_common_patterns(circuit)
        
        return circuit

    def _cancel_adjacent_gates(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Cancel adjacent gates that are inverses of each other."""
        # Create a new circuit with the same number of qubits and classical bits
        # but without copying registers to avoid name conflicts
        optimized_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        
        # Skip copying registers since that's causing the name conflict error
        
        # Extract instructions from the circuit
        instructions = list(circuit.data)
        i = 0
        
        # Process instructions sequentially
        while i < len(instructions):
            if i + 1 < len(instructions):
                curr_instr = instructions[i]
                next_instr = instructions[i + 1]
                
                # Check if gates act on the same qubits
                curr_qubits = curr_instr.qubits
                next_qubits = next_instr.qubits
                
                if curr_qubits == next_qubits:
                    # Check if gates are self-inverses (H, X, Y, Z)
                    if self._are_self_inverses(curr_instr, next_instr):
                        # Skip both gates (they cancel out)
                        i += 2
                        continue
                    # Check if gates are inverse rotation gates (e.g., Rx(θ) followed by Rx(-θ))
                    elif self._are_inverse_rotations(curr_instr, next_instr):
                        # Skip both gates
                        i += 2
                        continue
                
                # If no cancellation, add the current gate
                optimized_circuit.append(curr_instr.operation, curr_instr.qubits, curr_instr.clbits)
                i += 1
            else:
                # Add the last gate if there's no next gate to compare
                optimized_circuit.append(instructions[i].operation, instructions[i].qubits, instructions[i].clbits)
                i += 1
        
        return optimized_circuit

    def _are_self_inverses(self, instr1, instr2):
        """Check if instructions are self-inverse gates of the same type."""
        # Self-inverse gates: H, X, Y, Z, CNOT 
        self_inverse_gates = ['h', 'x', 'y', 'z', 'cx']
        
        gate1 = instr1.operation.name
        gate2 = instr2.operation.name
        
        # Check if both gates are self-inverses of the same type
        return gate1 == gate2 and gate1 in self_inverse_gates

    def _are_inverse_rotations(self, instr1, instr2):
        """Check if instructions are inverse rotation gates."""
        rotation_gates = ['rx', 'ry', 'rz', 'p', 'u1']
        
        gate1 = instr1.operation.name
        gate2 = instr2.operation.name
        
        # Check if gates are the same type of rotation
        if gate1 == gate2 and gate1 in rotation_gates:
            # Get rotation parameters
            if hasattr(instr1.operation, 'params') and hasattr(instr2.operation, 'params'):
                param1 = instr1.operation.params[0]
                param2 = instr2.operation.params[0]
                
                # Check if parameters sum to 0 (mod 2π for rotations)
                return abs(param1 + param2) < 1e-6 or abs(abs(param1 + param2) - 2*np.pi) < 1e-6
        
        return False

    def _same_qubits(self, instr1, instr2):
        """Check if two instructions act on the same qubits."""
        return instr1[1] == instr2[1]

    def _recognize_common_patterns(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Recognize and optimize common subcircuit patterns."""
        # Create a new circuit with the same number of qubits and classical bits
        # but without copying registers to avoid name conflicts
        optimized_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        
        # Skip copying registers since that's causing the name conflict error
        
        # Extract instructions from the circuit
        instructions = list(circuit.data)
        i = 0
        
        # Process instructions looking for patterns
        while i < len(instructions):
            # Pattern 1: H-CX-H -> Z-CX sequence (changing control/target basis)
            if i + 2 < len(instructions):
                # Check for H-CX-H pattern with H on the target qubit
                if (instructions[i].operation.name == 'h' and 
                    instructions[i+1].operation.name == 'cx' and 
                    instructions[i+2].operation.name == 'h' and
                    instructions[i].qubits[0] == instructions[i+2].qubits[0] and  # Same qubit for H gates
                    instructions[i].qubits[0] == instructions[i+1].qubits[1]):    # H is on CX target qubit
                    
                    # Replace with Z on control and CX
                    control = instructions[i+1].qubits[0]
                    target = instructions[i+1].qubits[1]
                    optimized_circuit.z(control)
                    optimized_circuit.cx(control, target)
                    i += 3
                    continue
            
            # Pattern 2: X-H-X -> Z-H sequence (Pauli X gates around Hadamard)
            if i + 2 < len(instructions):
                if (instructions[i].operation.name == 'x' and 
                    instructions[i+1].operation.name == 'h' and 
                    instructions[i+2].operation.name == 'x' and
                    instructions[i].qubits[0] == instructions[i+1].qubits[0] and  # Same qubit for X and H
                    instructions[i].qubits[0] == instructions[i+2].qubits[0]):    # Same qubit for both X
                    
                    # Replace with Z and H
                    qubit = instructions[i].qubits[0]
                    optimized_circuit.z(qubit)
                    optimized_circuit.h(qubit)
                    i += 3
                    continue
                    
            # Pattern 3: CNOT with target rotation - commutation relations
            if i + 1 < len(instructions):
                if (instructions[i].operation.name == 'cx' and 
                    instructions[i+1].operation.name in ['rz', 'p', 'u1'] and
                    instructions[i].qubits[1] == instructions[i+1].qubits[0]):  # Rotation is on CNOT target
                    
                    # Commute the rotation through CNOT (add control rotation)
                    control = instructions[i].qubits[0]
                    target = instructions[i].qubits[1]
                    rotation = instructions[i+1].operation.name
                    param = instructions[i+1].operation.params[0]
                    
                    # Add rotation to control qubit
                    if rotation == 'rz':
                        optimized_circuit.rz(param, control)
                    elif rotation in ['p', 'u1']:
                        optimized_circuit.p(param, control)
                    
                    # Add the CNOT gate
                    optimized_circuit.cx(control, target)
                    
                    # Add the original rotation on target
                    if rotation == 'rz':
                        optimized_circuit.rz(param, target)
                    elif rotation in ['p', 'u1']:
                        optimized_circuit.p(param, target)
                    
                    i += 2
                    continue
            
            # Pattern 4: Two Hadamards in a row become identity
            if i + 1 < len(instructions):
                if (instructions[i].operation.name == 'h' and 
                    instructions[i+1].operation.name == 'h' and
                    instructions[i].qubits[0] == instructions[i+1].qubits[0]):  # Same qubit
                    
                    # Skip both Hadamards (they cancel out to identity)
                    i += 2
                    continue
                    
            # If no pattern recognized, add the current instruction
            optimized_circuit.append(instructions[i].operation, instructions[i].qubits, instructions[i].clbits)
            i += 1
        
        return optimized_circuit

    def estimate_resources(self, circuit: QuantumCircuit) -> Dict[str, Union[int, float]]:
        """
        Estimate the resources required by a circuit.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Dictionary with resource estimates
        """
        num_qubits = circuit.num_qubits
        depth = circuit.depth()
        gate_counts = circuit.count_ops()
        
        # Count specific gate types
        cx_count = gate_counts.get('cx', 0)
        rotation_count = sum(gate_counts.get(g, 0) for g in ['rx', 'ry', 'rz'])
        
        # Estimate execution time (very approximate)
        # Assuming typical gate times in microseconds
        single_gate_time = 0.1  # microseconds
        two_qubit_gate_time = 1.0  # microseconds
        
        # Very rough estimate
        time_estimate = (
            sum(count for gate, count in gate_counts.items() if gate not in ['cx', 'cz', 'swap']) * single_gate_time +
            sum(count for gate, count in gate_counts.items() if gate in ['cx', 'cz', 'swap']) * two_qubit_gate_time
        )
        
        return {
            'num_qubits': num_qubits,
            'depth': depth,
            'gate_count': sum(gate_counts.values()),
            'cx_count': cx_count,
            'rotation_count': rotation_count,
            'estimated_time_us': time_estimate
        }
