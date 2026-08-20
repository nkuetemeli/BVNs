import random

from bvn_relu_layers import *
from utils import *
from utils_relu import *
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError, phase_damping_error


class BVNModelClAct:
    def __init__(self, n_qubits_input, in_dim, activation='Sigmoid', interference='Hadamard', standard_BV=True):
        self.hidden_layer = BVLayer(in_dim=in_dim, n_units=2, n_qubits_weight=1, n_qubits_output=1, start_wire=in_dim*n_qubits_input, activation=activation, activation_params={'threshold': 16, 'modulo': 2})
        self.output_layer = BVLayer(in_dim=2, n_units=1, n_qubits_weight=1, n_qubits_output=1, start_wire=self.hidden_layer.end_wire, activation=activation, activation_params={'threshold': 1, 'modulo': 2})
        self.interference = interference
        self.activation = activation
        self.standard_BV = standard_BV
        self.hidden_layer.describe()
        self.output_layer.describe()

        self.input_wires = [list(range(i*n_qubits_input, (i+1)*n_qubits_input)) for i in range(in_dim)]
        self._input_wires = [w for sublist in self.input_wires for w in sublist]
        self.weight_wires = self.hidden_layer._weight_wires + self.output_layer._weight_wires
        self.ancilla_wires = [self.output_layer.end_wire, self.output_layer.end_wire+1]
        self.interference_wires = [*self.input_wires] + ([*self.hidden_layer.activation_wires, self.output_layer._activation_wires] if not self.standard_BV else [])
        self._meas = None

    def circuit(self, state_vector, inputs):
        # Superposition of inputs
        qml.StatePrep(state_vector, wires=self._input_wires)

        # Ancilla preparation
        qml.X(wires=self.ancilla_wires[0])
        qml.Hadamard(wires=self.ancilla_wires[0])

        # Black-box call
        for input in inputs:
            control_values = [int(x) for x in input]
            qml.ctrl(qml.X, control=self._input_wires, control_values=control_values)(wires=self.ancilla_wires[0])

        # Ancilla un-computation
        qml.Hadamard(wires=self.ancilla_wires[0])
        qml.X(wires=self.ancilla_wires[0])

        if not self.standard_BV:
            # Superposition of hidden and outpout weights
            for wire in self.hidden_layer._weight_wires + self.output_layer._weight_wires:
                qml.Hadamard(wire)

            # Hidden and output activations
            self.hidden_layer.forward(self.input_wires, self.ancilla_wires)
            self.output_layer.forward(self.hidden_layer.activation_wires, self.ancilla_wires)

        # Interference
        for wires in self.interference_wires:
            if self.interference == 'Hadamard':
                for wire in wires:
                    qml.Hadamard(wire)
            if self.interference == 'Fourier':
                qml.QFT(wires=wires)
            if self.interference == 'Chebyshev':
                Chebyshev(ancilla_wire=self.ancilla_wires[0], wires=wires)

        return qml.counts()


    def run(self, dev, state_vector, inputs, shots, noise=False):
        if noise:
            circuit = qml.transforms.decompose(
                self.circuit, gate_set={"Hadamard", "PauliX", "PauliY", "PauliZ", "RX", "RY", "RZ", "CNOT", "CZ", },
            )
        else:
            circuit = self.circuit
        qnode = qml.QNode(lambda: circuit(state_vector, inputs), dev)
        qnode = qml.set_shots(qnode, shots=shots)
        return qnode()

    def compute_chi_matrix(self, X):
        """
        Vectorised χ-matrix computation for BVModelClAct or BVModelMask.
        X: (N, in_dim) input points
        """
        input_bits = self._meas["input_bits"]  # (M, in_dim, n_qubits_input)
        hidden_bits = self._meas.get("hidden_bits")  # (M, M_hidden, n_qubits_hidden)
        output_bits = self._meas.get("output_bits")  # (M, M_out, n_qubits_out)
        W_hidden = self._meas.get("W_hidden")  # (M_hidden, in_dim)
        W_out = self._meas.get("W_out")  # (M_out, M_hidden)

        N = X.shape[0]
        M = input_bits.shape[0]

        chi = np.ones((N, M), dtype=float)

        # --- Input interference ---
        for dim in range(input_bits.shape[1]):
            x_dim = X[:, dim]  # (N,)
            bits_dim = input_bits[:, dim, :]  # (M, n_qubits_input)
            chi *= interference_fn_vec(self.interference, x_dim, bits_dim)

        if not self.standard_BV and hidden_bits is not None:
            # --- Hidden activations ---
            h_raw = X @ W_hidden.T  # (N, M_hidden)
            h = np.zeros_like(h_raw)
            for i, unit in enumerate(self.hidden_layer.units):
                h[:, i] = unit.activation_fn(h_raw[:, i])

            # Vectorized over hidden units
            for i in range(hidden_bits.shape[1]):
                bits_h = hidden_bits[:, i, :]  # (M, n_qubits_hidden)
                chi *= interference_fn_vec(self.interference, h[:, i], bits_h)  # (N,M)

            # --- Output activations ---
            o_raw = h @ W_out.T  # (N, M_out)
            o = np.zeros_like(o_raw)
            for i, unit in enumerate(self.output_layer.units):
                o[:, i] = unit.activation_fn(o_raw[:, i])

            # Vectorized over output units
            for i in range(output_bits.shape[1]):
                bits_o = output_bits[:, i, :]  # (M, n_qubits_out)
                chi *= interference_fn_vec(self.interference, o[:, i], bits_o)  # (N,M)

        return chi

    def get_function_from_counts(self, counts, X, y, gram_scalar=1.):
        bitstrings = np.array([list(map(int, b)) for b in counts.keys()])

        # --- extract registers ---
        input_bits = np.stack([bitstrings[:, w] for w in self.input_wires], axis=1)
        hidden_bits = np.stack([bitstrings[:, unit.out_wires] for unit in self.hidden_layer.units], axis=1)
        output_bits = np.stack([bitstrings[:, unit.out_wires] for unit in self.output_layer.units], axis=1)
        W_hidden = np.array([[bitstrings[b, w] for w in unit.weight_wires] for b, unit in enumerate(self.hidden_layer.units)], dtype=int).squeeze(-1)
        W_out = np.array([[bitstrings[b, w] for w in unit.weight_wires] for b, unit in enumerate(self.output_layer.units)], dtype=int).squeeze(-1)

        self._meas = {
            "input_bits": input_bits,
            "hidden_bits": hidden_bits,
            "output_bits": output_bits,
            "W_hidden": W_hidden,
            "W_out": W_out
        }

        # Vectorised χ-matrix
        chi_eval = self.compute_chi_matrix(X)

        # classical linear solve
        results = basis_functions_from_matrix(chi_eval, y, gram_scalar)
        indices = np.abs(results['coeffs']).argsort()[::-1]
        results['coeffs'] = results['coeffs'][indices]
        self._meas["input_bits"] = input_bits[indices]
        self._meas["hidden_bits"] = hidden_bits[indices]
        self._meas["output_bits"] = output_bits[indices]
        return results


    def predict(self, X_new, results):
        chi = self.compute_chi_matrix(X_new)
        return chi @ results["coeffs"] + results["bias"]


if __name__ == '__main__':
    # Generate and visualize datasets
    shapes = {1: 'moons', 2: 'blobs', 3: 'sk_circles', 4: 'circle', 5: 'cross', 6: 'checkerboard',
              7: 'diamond', 8: 'h_stripes', 9: 'v_stripes', 10: 'spiral', 11: 'permutation', 12: 'relu', 13: 'random',
              20: 'penguins',
              21: 'iris',
              22: '1d'}
    shape = shapes[3]
    n_qubits = 4
    percentage = 75
    dummy_fill = (0, 0)

    interferences = {0: 'Hadamard', 1: 'Fourier', 2: 'Chebyshev',}
    interference = interferences[2]
    activation = 'ReLU'
    modes = {0: None, 1: 'noise', 2: 'random'}
    mode = modes[0]

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.005, 1), ["h", "x", "rz", "ry", "u"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx", ])
    noise_model.add_all_qubit_readout_error(ReadoutError([[0.98, 0.02], [0.02, 0.98]]))

    # Visualise training data
    scale = 2 ** n_qubits - 1
    X, y, X_full, y_full, _ = load_dataset(name=shape, scale=scale, percentage=percentage)
    X_filled, y_filled = fill_hyper_grid(X, y, scale, dummy_fill=dummy_fill)
    plot(X, y, X_full=X_full, y_full=y_full, y_full_pred=y_full, vmin=y.min(), vmax=y.max())

    # Oracle encoding in noise simulation, on binary classification only, otherwise amplitude encoding
    x_list = [''.join([np.binary_repr(a, width=n_qubits) for a in x]) for x in X_filled]
    amplitude_encoding = (shape in ['penguins', 'iris', '1d'] or not (mode == 'noise' and shape not in ['penguins', 'iris', '1d']))

    if amplitude_encoding:
        state_vector = uniform_state(len(x_list[0]), x_list, y=y_filled)
        inputs = []
    else:
        state_vector = uniform_state(len(x_list[0]), x_list, y=None)
        inputs = [x for x, label in zip(x_list, y_filled) if label == 1]

    for standard_BV in [True, False]:
        bv_model = BVNModelClAct(n_qubits_input=n_qubits, in_dim=X.shape[1], activation=activation, interference=interference, standard_BV=standard_BV)
        if mode is None:
            dev = qml.device('lightning.qubit', wires=bv_model.ancilla_wires[-1] + 1)
            counts = bv_model.run(dev, state_vector, inputs, shots=100)
        if mode == 'noise':
            dev = qml.device("qiskit.aer", wires=bv_model.ancilla_wires[-1] + 1, backend="aer_simulator", noise_model=noise_model)
            counts = bv_model.run(dev, state_vector, inputs, shots=100, noise=True)
        if mode == 'random':
            dev = qml.device('lightning.qubit', wires=bv_model.ancilla_wires[-1] + 1)
            counts = bv_model.run(dev, state_vector, inputs, shots=100)
            if standard_BV:
                l_effective = bv_model._input_wires[-1] + 1
                counts = {"".join(random.choices(["0", "1"], k=l_effective) + ['0'] * (len(x) - l_effective)): 0 for x in counts}
            else:
                counts = {"".join(random.choices(["0", "1"], k=len(x))): 0 for x in counts}

        if n_qubits < 5:
            qml.draw_mpl(bv_model.circuit)(state_vector, inputs=inputs)
            plt.show()

        results = bv_model.get_function_from_counts(counts, X, y, gram_scalar=.1)
        title = f'{"Std." if standard_BV else "Gen."} BVN'

        # Results reconstruction
        y_pred = bv_model.predict(X, results)
        y_full_pred = bv_model.predict(X_full, results)

        plot(X, y_pred, X_full=X_full, y_full=y_full, y_full_pred=y_full_pred, accuracy=round_to_labels_acc(y_full_pred, y_full, y_full), vmin=y.min(), vmax=y.max(),  title=title)


        # Full domain, real
        X_full_new = np.random.uniform(0, 2**n_qubits-1, (1024, X.shape[1]))
        y_pred_new = bv_model.predict(X_full_new, results)
        plot(X_full_new, y_pred_new, vmin=y.min(), vmax=y.max(), title=title)

