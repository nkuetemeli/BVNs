import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane.operation import Operation


import numpy as np

def uniform_state(n, bitstrings, y=None):
    state_vector = np.zeros(2 ** n, dtype=complex)

    if y is None:
        # Uniform superposition
        for bitstring in bitstrings:
            index = int(bitstring, 2)
            state_vector[index] = 1
    else:
        y = np.array(y, dtype=float)
        if len(y) != len(bitstrings):
            raise ValueError("Length of y must match number of bitstrings.")
        # Encode y as amplitudes
        for bitstring, val in zip(bitstrings, y):
            index = int(bitstring, 2)
            state_vector[index] = val
    # Normalize the state vector
    state_vector /= np.linalg.norm(state_vector)
    return state_vector

def bits_to_int(bits):
    """
    bits: array with shape (..., n_bits), last axis = bit dimension
    returns: array with shape (...) with the binary integer.
    """
    bits = np.asarray(bits)
    n_bits = bits.shape[-1]
    weights = (1 << np.arange(n_bits)[::-1])
    return np.tensordot(bits, weights, axes=([-1], [0]))


def interference_fn_vec(interference, a, bits):
    """
    Vectorised interference computation.

    Parameters:
    - a: array of real values (N, ...)  # input values
    - bits: array of integer bits (M, n_bits)

    Returns:
    - interference matrix of shape (N, M)
    """
    a = np.asarray(a)
    bits = np.asarray(bits)
    n_bits = bits.shape[-1]

    if interference == "Hadamard":
        # Floor input and convert to bit vectors
        a_int = np.floor(a).astype(int)  # shape (N,)
        a_bits = ((a_int[..., None] >> np.arange(n_bits)[::-1]) & 1)  # (N, n_bits)
        # Dot product modulo 2
        phase = np.sum(a_bits[:, None, :] * bits[None, :, :], axis=-1) % 2  # (N, M)
        result = (-1.0) ** phase

    # For Fourier & Chebyshev, treat a as real
    bits_int = bits_to_int(bits)  # shape (M,)

    if interference == "Fourier":
        result = np.exp(1j * 2 * np.pi * (a[:, None] * bits_int[None, :]) / (2**n_bits))
    if interference == "Chebyshev":
        result = np.cos(bits_int[None, :] * np.pi * (2*a[:, None] + 1) / (2**(n_bits+1)))

    return result.reshape(a.shape[0], bits.shape[0])


def basis_functions_from_matrix(chi_eval, y, gram_inversion=True, gram_scalar=.1):
    """
    Vectorised version of basis_functions(), expecting:

        chi_eval: (N_points, N_chi)   matrix of χ_i(x_j)
        y:        (N_points,)         target labels

    Returns:
        list of { "chi": NOT INCLUDED (because vectorised), "coeff": ... }
        PLUS an intercept term.
    """
    chi_eval = np.asarray(chi_eval)
    y = np.asarray(y)

    N_points, N_chi = chi_eval.shape
    Fhat = chi_eval.T @ y[:, None]   # (N_chi, 1)

    if gram_inversion:
        G = chi_eval.T @ chi_eval     # (N_chi, N_chi)
        G = G + gram_scalar * np.eye(N_chi)
        coeffs = (np.linalg.inv(G) @ Fhat).flatten()
        b = 0.0

    else:
        # Same logic as your original closed-form LS with intercept.
        XFhat = chi_eval @ Fhat            # (N_points, 1)
        F = y[:, None]
        ones = np.ones_like(F)

        den = (ones.T @ ones) * np.sum(XFhat ** 2) - np.sum(XFhat.T @ ones) ** 2
        s = XFhat.T @ ((ones.T @ ones) * F - (ones @ ones.T) @ F) / den if den > 1e-12 else 1.0
        b = ((ones.T @ F - s * ones.T @ XFhat) / (ones.T @ ones)).item()

        coeffs = (s * Fhat).flatten()

    # Return only coefficients; χ functions are now vectorised
    return {
        "coeffs": coeffs,     # shape (N_chi,)
        "bias": b
    }


class Chebyshev(qml.operation.Operation):

    def __init__(self, ancilla_wire, wires):
        # Combine the ancilla and working wires into one `all_wires` list
        all_wires = qml.wires.Wires(ancilla_wire) + qml.wires.Wires(wires)
        super().__init__(wires=all_wires)

    @staticmethod
    def compute_decomposition(*params, wires, **hyperparams):
        # Initialize the list of operations
        op_list = []

        # Extract ancilla wire and working wires
        n = len(wires) - 1
        ancilla_wire = wires[0]
        working_wires = wires[1:]

        # Apply Hadamard gate to the ancilla wire
        op_list.append(qml.Hadamard(wires=ancilla_wire))

        # Add CNOT ladder
        for i in range(0, n):
            op_list.append(qml.CNOT(wires=[ancilla_wire, working_wires[i]]))

        # Add QFT (use built-in QFT)
        op_list.append(qml.QFT(wires=[ancilla_wire] + working_wires))

        # Add U1 CRs
        op_list.append(qml.RZ(-np.pi * (2 ** n - 1) / (2 ** (n + 1)), wires=ancilla_wire))
        op_list.append(qml.PhaseShift(-np.pi / (2 ** (n + 1)), wires=ancilla_wire))

        # Add individual CRs
        for i in range(0, n):
            op_list.append(qml.RZ(np.pi / 2 ** (i + 2), wires=working_wires[i]))

        # Multi-controlled X (Permutation Step)
        for i, wire in enumerate(working_wires[::-1]):
            control_wires = [ancilla_wire] + working_wires[n - i:]
            op_list.append(qml.MultiControlledX(wires=control_wires + [wire]))

        # Add second CNOT ladder
        for i in range(0, n):
            op_list.append(qml.CNOT(wires=[ancilla_wire, working_wires[i]]))

        # Apply U2 transformations
        op_list.append(qml.RY(-np.pi / 2, wires=ancilla_wire))
        op_list.append(qml.PhaseShift(-np.pi / 2, wires=ancilla_wire))

        # Adjust phase
        for i in range(n):
            op_list.append(qml.PauliX(wires=working_wires[i]))

        # Controlled RX gate
        op_list.append(qml.ctrl(qml.RX(np.pi / 2, wires=ancilla_wire), control=working_wires))

        # Reset phase
        for i in range(n):
            op_list.append(qml.PauliX(wires=working_wires[i]))

        # Return the list of operations
        return op_list



def plot_counts_histogram(counts):
    """plots a histogram of quantum measurement results."""
    if isinstance(counts, list):
        counts = counts[0]

    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Measurement outcome")
    plt.ylabel("Counts")
    plt.title("Quantum Measurement Histogram")
    plt.xticks(rotation=90)  # Rotate labels for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()


def plot_cheb_basis(n):
    chebs = []

    fig, ax = plt.subplots(2 ** n, 2)
    fig.set_size_inches(9, 16)
    for j in range(2 ** n):
        cheb = []
        for k in range(2 ** n):
            tk = np.cos(k * np.pi * (2 * j + 1) / 2 ** (n + 1))
            if k == 0:
                tk /= 2 ** (n / 2)
            else:
                tk /= 2 ** ((n - 1) / 2)
            cheb.append(tk)
        cheb = np.array(cheb)
        chebs.append(cheb)
        ax[j, 0].bar(list(range(len(cheb))), cheb)
        ax[j, 1].bar(list(range(len(cheb))), cheb ** 2)
        ax[j, 0].set_title(f'j = {j}')
        ax[j, 1].set_title(f'j = {j}')
    plt.show()
    chebs = np.vstack(chebs)
    return chebs



if __name__ == '__main__':

    n = 3

    plot_cheb_basis(n)

    num_qubits = n + 1
    dev = qml.device("default.qubit", wires=num_qubits, shots=1024)


    @qml.qnode(dev)
    def circuit(n):
        Chebyshev(ancilla_wire=0, wires=list(range(1, n+1)))  # Apply the Chebyshev circuit
        return qml.counts(wires=range(1+n))  # Measure all non-ancilla qubits

    @qml.qnode(dev, interface="torch")
    def circuit2(n):
        Chebyshev(ancilla_wire=0, wires=list(range(1, n+1)))  # Apply the Chebyshev circuit
        return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in range(1, n+1)]))

    fig, ax = qml.draw_mpl(circuit)(n)
    plt.show()

    counts = circuit(n)
    if isinstance(counts, list):
        counts = counts[0]
    plot_counts_histogram(counts)

    print(circuit2(n))

