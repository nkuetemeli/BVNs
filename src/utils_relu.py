import matplotlib.pyplot as plt
import pennylane as qml


class CheapReLU(qml.operation.Operation):
    """Overflow-guarded reversible multiplication block.

    |x>|y>|0>|0> → |x>|y>|σ(xy)>|flag>
    σ(xy) = xy if no overflow, else 0.
    """

    num_wires = None
    grad_method = None  # non-differentiable

    def __init__(self, x1_wires, x2_wires, y1_wires, y2_wires, ancilla_wire, sigma_wires):

        all_wires = qml.wires.Wires(x1_wires + x2_wires + y1_wires + y2_wires + sigma_wires)

        self._hyperparameters = {
            'x1_wires': x1_wires,
            'x2_wires': x2_wires,
            'y1_wires': y1_wires,
            'y2_wires': y2_wires,
            'ancilla_wire': ancilla_wire,
            'sigma_wires': sigma_wires,
        }
        super().__init__(wires=all_wires)

    @staticmethod
    def compute_decomposition(wires, x1_wires, x2_wires, y1_wires, y2_wires, ancilla_wire, sigma_wires, ):
        op_list = []

        # Overflow detector
        op_list.append(
            qml.ctrl(qml.OutMultiplier(x_wires=x2_wires, y_wires=y2_wires, output_wires=ancilla_wire+x1_wires),
                     control=y1_wires, control_values=[1])
        )

        relu_precision_complement = len(ancilla_wire+x1_wires) - len(sigma_wires)
        control_wires = y1_wires + (ancilla_wire+x1_wires)[:relu_precision_complement]
        for x_wire, sigma_wire in zip((ancilla_wire+x1_wires)[relu_precision_complement:], sigma_wires):
            qml.ctrl(qml.CNOT, control=control_wires, control_values=[1] + [0]*relu_precision_complement)([x_wire, sigma_wire])

        control_wires = y1_wires + y2_wires + (ancilla_wire+x2_wires)[:relu_precision_complement]
        for x_wire, sigma_wire in zip((ancilla_wire+x2_wires)[relu_precision_complement:], sigma_wires):
            qml.ctrl(qml.CNOT, control=control_wires, control_values=[0, 1]+[0]*relu_precision_complement)([x_wire, sigma_wire])

        op_list.append(
            qml.adjoint(qml.ctrl(qml.OutMultiplier(x_wires=x2_wires, y_wires=y2_wires, output_wires=ancilla_wire+x1_wires),
                     control=y1_wires, control_values=[1]))
        )
        # Return the list of operations
        return op_list

def decode_counts(counts, x_len, y_len, sigma_len):
    """
    Decode PennyLane bitstring counts into integers: (x1, y1, x2, y2, sigma, flag)

    Args:
        counts (dict): PennyLane counts dictionary.
        x1_len, y1_len, x2_len, y2_len, sigma_len (int): number of qubits in each register.

    Returns:
        list of tuples: (x1, y1, x2, y2, sigma, flag, freq)
    """
    decoded = []
    for bitstring, freq in counts.items():
        # Bitstring is MSB -> LSB
        bits = bitstring

        # Extract slices
        start = 0
        x1_bits = bits[start:start + x_len]
        start += x_len
        x2_bits = bits[start:start + x_len]
        start += x_len
        y1_bits = bits[start:start + y_len]
        start += y_len
        y2_bits = bits[start:start + y_len]
        start += y_len
        sigma_bits = bits[start:start + sigma_len]
        start += sigma_len
        ancilla_bit = bits[start:start+1]

        # Convert to integers
        x1_val = int(x1_bits, 2)
        x2_val = int(x2_bits, 2)
        y1_val = int(y1_bits, 2)
        y2_val = int(y2_bits, 2)
        sigma_val = int(sigma_bits, 2)
        ancilla_val = ancilla_bit

        decoded.append((x1_val, x2_val, y1_val, y2_val, sigma_val, ancilla_val, freq))
    return decoded


if __name__ == "__main__":
    # Device
    n, k, l = 2, 1, 1
    total_wires = 2*n + 2*k + l + 1  # +1 for flag
    shots = 1024
    dev = qml.device("default.qubit", wires=total_wires)


    # Wire assignment
    x1_wires = list(range(n))
    x2_wires = list(range(n, n + n))
    y1_wires = list(range(n + n, n + n + k))
    y2_wires = list(range(n + n + k, n + n + k + k))
    sigma_wires = list(range(n + k + n + k, n + k + n + k + l))
    ancilla_wire = [total_wires - 1] # extra ancilla for accumulation
    relu_threshold = 2**l

    @qml.qnode(dev)
    def circuit():
        # Put each qubit in superposition for testing
        for wire in x1_wires + x2_wires + y1_wires + y2_wires:
            qml.Hadamard(wire)

        # Call the modular ReLU with x1y1 + x2y2
        CheapReLU(x1_wires, x2_wires, y1_wires, y2_wires, ancilla_wire, sigma_wires).decomposition()

        return qml.counts()
    circuit = qml.set_shots(circuit, shots=shots)

    # Visualize the circuit
    print(qml.draw_mpl(circuit)())
    plt.show()

    # Run the circuit
    counts = circuit()
    print("Final counts:", counts)

    # Decode
    # Decode
    decoded = decode_counts(counts, n, k, l)
    print("\nDecoded results (x1, y1, x2, y2, σ, a, flag, freq):")
    for x1, x2, y1, y2, s, a, freq in decoded:
        true_val = (x1 * y1 + x2 * y2)
        true_val = true_val if true_val <= relu_threshold-1 else 0
        print(f"x1={x1:02d}, y1={y1:02d}, x2={x2:02d}, y2={y2:02d}, a={a}, σ={s:02d}, true={true_val:02d}")
        if s != true_val:
            print("Failed")