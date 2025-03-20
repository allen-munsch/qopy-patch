#!/usr/bin/env python
"""
Circuit Inspection Script - Visualize and analyze quantum circuits in detail.

This script creates and visualizes quantum circuits used in the JIT system,
allowing detailed inspection of their structure and components.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, plot_state_city, plot_bloch_multivector
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
import os

# Ensure output directory exists
os.makedirs('circuit_inspection', exist_ok=True)

def inspect_hadamard_circuit(num_qubits=3):
    """Detailed inspection of the Hadamard transform circuit."""
    print(f"\n{'='*50}")
    print(f"INSPECTING {num_qubits}-QUBIT HADAMARD TRANSFORM CIRCUIT")
    print(f"{'='*50}")
    
    # Create the circuit
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Apply Hadamard gates to all qubits
    for i in range(num_qubits):
        qc.h(i)
    
    # Save a version without measurements for statevector simulation
    qc_no_measure = qc.copy()
    
    # Add measurements for the standard circuit
    qc.measure(range(num_qubits), range(num_qubits))
    
    # Print circuit details
    print("\nCircuit Details:")
    print(f"Number of qubits: {qc.num_qubits}")
    print(f"Number of classical bits: {qc.num_clbits}")
    print(f"Circuit depth: {qc.depth()}")
    print(f"Total number of operations: {len(qc.data)}")
    
    # Count gate types
    gate_counts = {}
    for instr in qc.data:
        gate_name = instr[0].name
        if gate_name in gate_counts:
            gate_counts[gate_name] += 1
        else:
            gate_counts[gate_name] = 1
    
    print("\nGate composition:")
    for gate, count in gate_counts.items():
        print(f"  - {gate}: {count}")
    
    # Draw circuit in different styles
    print("\nSaving circuit visualizations...")
    
    # Text representation
    with open('circuit_inspection/hadamard_text.txt', 'w') as f:
        f.write(qc.draw(output='text').single_string())
    print("  - Text representation saved to 'circuit_inspection/hadamard_text.txt'")
    
    # Standard matplotlib visualization
    fig_mpl = qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
    plt.title(f"{num_qubits}-Qubit Hadamard Transform Circuit")
    plt.tight_layout()
    plt.savefig('circuit_inspection/hadamard_circuit.png', dpi=300)
    plt.close()
    print("  - Standard visualization saved to 'circuit_inspection/hadamard_circuit.png'")
    
    # Simulate the circuit (without measurement)
    simulator = Aer.get_backend('statevector_simulator')
    result = simulator.run(qc_no_measure).result()
    statevector = result.get_statevector()
    
    # Plot the quantum state
    state_fig = plot_state_city(statevector)
    plt.title(f"Quantum State After {num_qubits}-Qubit Hadamard Transform")
    plt.tight_layout()
    plt.savefig('circuit_inspection/hadamard_state.png', dpi=300)
    plt.close()
    print("  - Quantum state visualization saved to 'circuit_inspection/hadamard_state.png'")
    
    # For small number of qubits, show Bloch sphere representation
    if num_qubits <= 5:  # Limit for reasonable visualization
        bloch_fig = plot_bloch_multivector(statevector)
        plt.savefig('circuit_inspection/hadamard_bloch.png', dpi=300)
        plt.close()
        print("  - Bloch sphere visualization saved to 'circuit_inspection/hadamard_bloch.png'")
    
    # Execute the circuit with measurements
    qasm_simulator = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(qc, qasm_simulator)
    counts_result = qasm_simulator.run(transpiled_circuit, shots=4096).result()
    counts = counts_result.get_counts(qc)
    
    # Plot histogram of results
    hist_fig = plot_histogram(counts)
    plt.title(f"Measurement Outcomes of {num_qubits}-Qubit Hadamard Transform")
    plt.tight_layout()
    plt.savefig('circuit_inspection/hadamard_histogram.png', dpi=300)
    plt.close()
    print("  - Measurement histogram saved to 'circuit_inspection/hadamard_histogram.png'")
    
    # Calculate and display theoretical probabilities
    print("\nTheoretical analysis:")
    print(f"  - Expected probability per state: {1/2**num_qubits:.6f}")
    print(f"  - Number of possible states: {2**num_qubits}")
    
    # Check if measurements match theoretical distribution
    total_shots = sum(counts.values())
    chi_square = sum((count - total_shots/2**num_qubits)**2 / (total_shots/2**num_qubits) for count in counts.values())
    print(f"  - Chi-square test statistic: {chi_square:.2f}")
    print(f"  - Average deviation from expected: {np.sqrt(chi_square/2**num_qubits):.2f} sigma")
    
    return qc

def inspect_qft_circuit(num_qubits=3):
    """Detailed inspection of the Quantum Fourier Transform circuit."""
    print(f"\n{'='*50}")
    print(f"INSPECTING {num_qubits}-QUBIT QUANTUM FOURIER TRANSFORM CIRCUIT")
    print(f"{'='*50}")
    
    # Create the circuit
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Prepare an initial state (|1⟩)
    qc.x(0)
    
    # Add QFT circuit
    for i in range(num_qubits):
        qc.h(i)
        for j in range(i + 1, num_qubits):
            # Controlled phase rotation
            qc.cp(2 * np.pi / 2**(j-i+1), j, i)
    
    # Swap qubits for the proper QFT ordering
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - i - 1)
    
    # Save a version without measurements for statevector simulation
    qc_no_measure = qc.copy()
    
    # Add measurements
    qc.measure(range(num_qubits), range(num_qubits))
    
    # Print circuit details
    print("\nCircuit Details:")
    print(f"Number of qubits: {qc.num_qubits}")
    print(f"Number of classical bits: {qc.num_clbits}")
    print(f"Circuit depth: {qc.depth()}")
    print(f"Total number of operations: {len(qc.data)}")
    
    # Count gate types
    gate_counts = {}
    for instr in qc.data:
        gate_name = instr[0].name
        if gate_name in gate_counts:
            gate_counts[gate_name] += 1
        else:
            gate_counts[gate_name] = 1
    
    print("\nGate composition:")
    for gate, count in sorted(gate_counts.items()):
        print(f"  - {gate}: {count}")
    
    # Draw circuit in different styles
    print("\nSaving circuit visualizations...")
    
    # Text representation
    with open('circuit_inspection/qft_text.txt', 'w') as f:
        f.write(qc.draw(output='text').single_string())
    print("  - Text representation saved to 'circuit_inspection/qft_text.txt'")
    
    # Standard matplotlib visualization
    fig_mpl = qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
    plt.title(f"{num_qubits}-Qubit Quantum Fourier Transform Circuit")
    plt.tight_layout()
    plt.savefig('circuit_inspection/qft_circuit.png', dpi=300)
    plt.close()
    print("  - Standard visualization saved to 'circuit_inspection/qft_circuit.png'")
    
    # Simulate the circuit (without measurement)
    simulator = Aer.get_backend('statevector_simulator')
    result = simulator.run(qc_no_measure).result()
    statevector = result.get_statevector()
    
    # Plot the quantum state
    state_fig = plot_state_city(statevector)
    plt.title(f"Quantum State After QFT of |1⟩")
    plt.tight_layout()
    plt.savefig('circuit_inspection/qft_state.png', dpi=300)
    plt.close()
    print("  - Quantum state visualization saved to 'circuit_inspection/qft_state.png'")
    
    # Print amplitudes for analysis
    print("\nStatevector Analysis:")
    print("State | Amplitude | Probability")
    print("-" * 40)
    for i, amplitude in enumerate(statevector):
        if abs(amplitude) > 1e-10:  # Filter out near-zero amplitudes
            binary = format(i, f'0{num_qubits}b')
            prob = abs(amplitude)**2
            phase = np.angle(amplitude)
            print(f"|{binary}⟩ | {amplitude:.4f} | {prob:.4f} | Phase: {phase:.4f}")
    
    # Execute the circuit with measurements
    qasm_simulator = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(qc, qasm_simulator)
    counts_result = qasm_simulator.run(transpiled_circuit, shots=4096).result()
    counts = counts_result.get_counts(qc)
    
    # Plot histogram of results
    hist_fig = plot_histogram(counts)
    plt.title(f"Measurement Outcomes of QFT on |1⟩")
    plt.tight_layout()
    plt.savefig('circuit_inspection/qft_histogram.png', dpi=300)
    plt.close()
    print("  - Measurement histogram saved to 'circuit_inspection/qft_histogram.png'")
    
    return qc

def inspect_grover_circuit(num_qubits=3, target_state='101'):
    """Detailed inspection of Grover's search algorithm circuit."""
    print(f"\n{'='*50}")
    print(f"INSPECTING {num_qubits}-QUBIT GROVER'S SEARCH CIRCUIT (Target: |{target_state}⟩)")
    print(f"{'='*50}")
    
    # Convert target state from binary to int
    target_int = int(target_state, 2)
    
    # Create the circuit
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Apply H gates to put qubits in superposition
    for i in range(num_qubits):
        qc.h(i)
    
    # Oracle for marking the target state
    # Convert target state to binary and apply X gates where needed
    for i in range(num_qubits):
        if target_state[i] == '0':
            qc.x(i)
    
    # Apply multi-controlled Z gate
    qc.h(num_qubits-1)
    qc.mcx(list(range(num_qubits-1)), num_qubits-1)
    qc.h(num_qubits-1)
    
    # Undo X gates
    for i in range(num_qubits):
        if target_state[i] == '0':
            qc.x(i)
    
    # Diffusion operator (amplification)
    for i in range(num_qubits):
        qc.h(i)
    
    for i in range(num_qubits):
        qc.x(i)
    
    # Apply multi-controlled Z gate for diffusion
    qc.h(num_qubits-1)
    qc.mcx(list(range(num_qubits-1)), num_qubits-1)
    qc.h(num_qubits-1)
    
    for i in range(num_qubits):
        qc.x(i)
    
    for i in range(num_qubits):
        qc.h(i)
    
    # Save a version without measurements for statevector simulation
    qc_no_measure = qc.copy()
    
    # Add measurements
    qc.measure(range(num_qubits), range(num_qubits))
    
    # Print circuit details
    print("\nCircuit Details:")
    print(f"Number of qubits: {qc.num_qubits}")
    print(f"Number of classical bits: {qc.num_clbits}")
    print(f"Circuit depth: {qc.depth()}")
    print(f"Total number of operations: {len(qc.data)}")
    
    # Count gate types
    gate_counts = {}
    for instr in qc.data:
        gate_name = instr[0].name
        if gate_name in gate_counts:
            gate_counts[gate_name] += 1
        else:
            gate_counts[gate_name] = 1
    
    print("\nGate composition:")
    for gate, count in sorted(gate_counts.items()):
        print(f"  - {gate}: {count}")
    
    # Draw circuit in different styles
    print("\nSaving circuit visualizations...")
    
    # Text representation
    with open('circuit_inspection/grover_text.txt', 'w') as f:
        f.write(qc.draw(output='text').single_string())
    print("  - Text representation saved to 'circuit_inspection/grover_text.txt'")
    
    # Standard matplotlib visualization
    fig_mpl = qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
    plt.title(f"{num_qubits}-Qubit Grover's Search Circuit (Target: |{target_state}⟩)")
    plt.tight_layout()
    plt.savefig('circuit_inspection/grover_circuit.png', dpi=300)
    plt.close()
    print("  - Standard visualization saved to 'circuit_inspection/grover_circuit.png'")
    
    # Simulate the circuit (without measurement)
    simulator = Aer.get_backend('statevector_simulator')
    result = simulator.run(qc_no_measure).result()
    statevector = result.get_statevector()
    
    # Plot the quantum state
    state_fig = plot_state_city(statevector)
    plt.title(f"Quantum State After Grover's Search for |{target_state}⟩")
    plt.tight_layout()
    plt.savefig('circuit_inspection/grover_state.png', dpi=300)
    plt.close()
    print("  - Quantum state visualization saved to 'circuit_inspection/grover_state.png'")
    
    # Print amplitudes for the target state
    target_amplitude = statevector[target_int]
    target_probability = abs(target_amplitude)**2
    print(f"\nTarget state |{target_state}⟩ analysis:")
    print(f"  - Amplitude: {target_amplitude:.6f}")
    print(f"  - Probability: {target_probability:.6f}")
    
    # Calculate theoretical success probability for 1 Grover iteration
    n = num_qubits
    N = 2**n
    theoretical_prob = np.sin((2*1 + 1) * np.arcsin(1/np.sqrt(N)))**2
    print(f"  - Theoretical probability: {theoretical_prob:.6f}")
    
    # Execute the circuit with measurements
    qasm_simulator = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(qc, qasm_simulator)
    counts_result = qasm_simulator.run(transpiled_circuit, shots=4096).result()
    counts = counts_result.get_counts(qc)
    
    # Plot histogram of results
    hist_fig = plot_histogram(counts)
    plt.title(f"Measurement Outcomes of Grover's Search for |{target_state}⟩")
    plt.tight_layout()
    plt.savefig('circuit_inspection/grover_histogram.png', dpi=300)
    plt.close()
    print("  - Measurement histogram saved to 'circuit_inspection/grover_histogram.png'")
    
    return qc

def inspect_binary_function_circuit(num_qubits=3, function_type='parity'):
    """Detailed inspection of binary function evaluation circuit."""
    print(f"\n{'='*50}")
    print(f"INSPECTING {num_qubits}-QUBIT BINARY FUNCTION EVALUATION CIRCUIT ({function_type})")
    print(f"{'='*50}")
    
    # Create the circuit with input qubits and 1 output qubit
    qc = QuantumCircuit(num_qubits + 1, num_qubits)
    
    # Put input qubits in superposition
    for i in range(num_qubits):
        qc.h(i)
    
    # Initialize output qubit
    qc.x(num_qubits)
    qc.h(num_qubits)
    
    # Implement function
    if function_type == 'parity':
        print("\nImplementing parity function: f(x) = 1 if number of 1s is odd, else 0")
        for i in range(num_qubits):
            qc.cx(i, num_qubits)
    elif function_type == 'constant_one':
        print("\nImplementing constant function: f(x) = 1 for all x")
        # For constant 1, do nothing to output qubit
        pass
    elif function_type == 'constant_zero':
        print("\nImplementing constant function: f(x) = 0 for all x")
        # Flip output qubit
        qc.z(num_qubits)
    elif function_type == 'balanced':
        print("\nImplementing balanced function: f(x) = x_0 (value of first bit)")
        qc.cx(0, num_qubits)
    
    # Save a version without measurements for statevector simulation
    qc_no_measure = qc.copy()
    
    # Measure input qubits
    qc.measure(range(num_qubits), range(num_qubits))
    
    # Print circuit details
    print("\nCircuit Details:")
    print(f"Number of qubits: {qc.num_qubits}")
    print(f"Number of classical bits: {qc.num_clbits}")
    print(f"Circuit depth: {qc.depth()}")
    print(f"Total number of operations: {len(qc.data)}")
    
    # Count gate types
    gate_counts = {}
    for instr in qc.data:
        gate_name = instr[0].name
        if gate_name in gate_counts:
            gate_counts[gate_name] += 1
        else:
            gate_counts[gate_name] = 1
    
    print("\nGate composition:")
    for gate, count in sorted(gate_counts.items()):
        print(f"  - {gate}: {count}")
    
    # Draw circuit in different styles
    print("\nSaving circuit visualizations...")
    
    # Text representation
    with open(f'circuit_inspection/binary_function_{function_type}_text.txt', 'w') as f:
        f.write(qc.draw(output='text').single_string())
    print(f"  - Text representation saved to 'circuit_inspection/binary_function_{function_type}_text.txt'")
    
    # Standard matplotlib visualization
    fig_mpl = qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
    plt.title(f"{num_qubits}-Qubit Binary Function Evaluation ({function_type})")
    plt.tight_layout()
    plt.savefig(f'circuit_inspection/binary_function_{function_type}_circuit.png', dpi=300)
    plt.close()
    print(f"  - Standard visualization saved to 'circuit_inspection/binary_function_{function_type}_circuit.png'")
    
    # Simulate the circuit (without measurement)
    simulator = Aer.get_backend('statevector_simulator')
    result = simulator.run(qc_no_measure).result()
    statevector = result.get_statevector()
    
    # Plot the quantum state
    state_fig = plot_state_city(statevector)
    plt.title(f"Quantum State After Binary Function Evaluation ({function_type})")
    plt.tight_layout()
    plt.savefig(f'circuit_inspection/binary_function_{function_type}_state.png', dpi=300)
    plt.close()
    print(f"  - Quantum state visualization saved to 'circuit_inspection/binary_function_{function_type}_state.png'")
    
    # Execute the circuit with measurements
    qasm_simulator = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(qc, qasm_simulator)
    counts_result = qasm_simulator.run(transpiled_circuit, shots=4096).result()
    counts = counts_result.get_counts(qc)
    
    # Plot histogram of results
    hist_fig = plot_histogram(counts)
    plt.title(f"Measurement Outcomes of Binary Function Evaluation ({function_type})")
    plt.tight_layout()
    plt.savefig(f'circuit_inspection/binary_function_{function_type}_histogram.png', dpi=300)
    plt.close()
    print(f"  - Measurement histogram saved to 'circuit_inspection/binary_function_{function_type}_histogram.png'")
    
    # For parity function, check the function output for each input
    if function_type == 'parity':
        print("\nVerifying parity function outputs:")
        print("Input | Expected Output | Measured Probability")
        print("-" * 50)
        for i in range(2**num_qubits):
            binary = format(i, f'0{num_qubits}b')
            expected = bin(i).count('1') % 2
            if binary in counts:
                prob = counts[binary] / sum(counts.values())
                print(f"|{binary}⟩ | {expected} | {prob:.4f}")
            else:
                print(f"|{binary}⟩ | {expected} | Not measured")
    
    return qc

# Main inspection function
def inspect_circuits():
    """Inspect all circuit types used in the quantum JIT system."""
    print("\nWelcome to the Quantum Circuit Inspector!")
    print("This tool will generate detailed visualizations and analysis of quantum circuits.")
    print("\nStarting circuit inspections...\n")
    
    # Inspect Hadamard transform circuit
    hadamard_circuit = inspect_hadamard_circuit(num_qubits=3)
    
    # Inspect QFT circuit
    qft_circuit = inspect_qft_circuit(num_qubits=3)
    
    # Inspect Grover's search circuit
    grover_circuit = inspect_grover_circuit(num_qubits=3, target_state='101')
    
    # Inspect binary function circuits
    binary_circuit = inspect_binary_function_circuit(num_qubits=3, function_type='parity')
    
    print(f"\n{'='*50}")
    print("CIRCUIT INSPECTION COMPLETE")
    print(f"{'='*50}")
    print("\nAll circuit inspections have been saved to the 'circuit_inspection' directory.")
    print("You can view the detailed circuit diagrams, statevectors, and measurement histograms there.")

if __name__ == "__main__":
    inspect_circuits()