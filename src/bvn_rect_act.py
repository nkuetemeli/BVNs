import random

from bvn_relu_layers import *
from utils import *
from utils_mask import *
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError, phase_damping_error


class BVNModelMask:
    def __init__(self, n_qubits_input, in_dim, mask_resolution=2, interference='Hadamard', standard_BV=True):
        self.n_qubits_input = n_qubits_input
        self.interference = interference
        self.standard_BV = standard_BV
        self.super_size = self.n_qubits_input - mask_resolution

        self.input_wires = np.array([list(range(i*n_qubits_input, (i+1)*n_qubits_input)) for i in range(in_dim)]).tolist()
        self._input_wires = [w for sublist in self.input_wires for w in sublist]
        self.mask_wires = (self._input_wires[-1] + 1 + np.array([list(range(i*mask_resolution, (i+1)*mask_resolution)) for i in range(in_dim)])).tolist()
        self._mask_wires = [w for sublist in self.mask_wires for w in sublist]
        self.superposition_wires = (self._mask_wires[-1] + np.array([1])).tolist()
        self.activations_wires = (self.superposition_wires[-1] + np.array([1])).tolist()
        self.ancilla_wires = (self.activations_wires[-1] + np.array([1])).tolist()

        self.interference_wires = self.input_wires + ([self.activations_wires] if not self.standard_BV else [])
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
            for wire in self._mask_wires + self.superposition_wires:
                qml.Hadamard(wires=wire)

            CheapMask(
                x_wires=self.input_wires,
                mask_wires=self.mask_wires,
                superposition_wire=self.superposition_wires,
                sigma_wire=self.activations_wires,
            )

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


    def run(self, dev, state_vector, inputs, shots):
        qnode = qml.QNode(lambda: self.circuit(state_vector, inputs), dev)
        qnode = qml.set_shots(qnode, shots=shots)
        return qnode()

    def compute_chi_matrix(self, X):
        input_bits = np.asarray(self._meas["input_bits"])  # (M,in_dim,n_bits)
        mask_bits = np.asarray(self._meas["mask_bits"])  # (M,in_dim,k_mask)
        superposition_bits = np.asarray(self._meas["superposition_bits"])  # (M,)
        output_bits = np.asarray(self._meas["output_bits"])  # (M,n_out)

        N = len(X)
        M = len(input_bits)
        chi = np.ones((N, M))

        # ---------------- INPUT INTERFERENCE ----------------
        for dim in range(input_bits.shape[1]):
            x_dim = X[:, dim]  # (N,)
            bits_dim = input_bits[:, dim, :]  # (M,n_bits)
            chi *= interference_fn_vec(self.interference, x_dim, bits_dim)

        # ---------------- GENERALIZED MODEL (mask) INTERFERENCE ----------------
        if not self.standard_BV:
            mask_width = mask_bits.shape[-1]

            # (M,in_dim)
            position = mask_bits.dot(1 << np.arange(mask_width)[::-1])

            # Compute masked values
            mask_flag = np.ones((N, M), dtype=bool)
            for dim in range(X.shape[1]):
                x_d = X[:, dim][:, None]
                p_d = position[:, dim][None, :]
                sp = superposition_bits.T

                masked = (x_d + p_d * 2**self.super_size + sp * 2**(self.super_size-1)) % (2 ** self.n_qubits_input)
                mask_flag &= (masked < 2 ** self.super_size)

            mask = mask_flag.astype(int)  # (N,M)
            chi *= interference_fn_vec(self.interference, mask, output_bits)

        return chi

    def get_function_from_counts(self, counts, X, y, gram_scalar=1.):
        bitstrings = np.array([list(map(int, b)) for b in counts.keys()])

        # extract all registers
        input_bits = np.stack([bitstrings[:, w] for w in self.input_wires], axis=1)
        mask_bits = np.stack([bitstrings[:, w] for w in self.mask_wires], axis=1)
        superposition_bits = bitstrings[:, self.superposition_wires]
        output_bits = bitstrings[:, self.activations_wires]

        self._meas = {
            "input_bits": input_bits,
            "mask_bits": mask_bits,
            "superposition_bits": superposition_bits,
            "output_bits": output_bits
        }

        # vectorised chi matrix
        chi_eval = self.compute_chi_matrix(X)

        # classical linear solve
        results = basis_functions_from_matrix(chi_eval, y, gram_scalar)
        indices = np.abs(results['coeffs']).argsort()[::-1]
        results['coeffs'] = results['coeffs'][indices]
        self._meas["input_bits"] = input_bits[indices]
        self._meas["mask_bits"] = mask_bits[indices]
        self._meas["superposition_bits"] = superposition_bits[indices]
        self._meas["output_bits"] = output_bits[indices]
        return results

    def predict(self, X_new, results):
        chi = self.compute_chi_matrix(X_new,)
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

    # Standard vs. Generalised BVN
    for standard_BV in [True, False]:
        bv_model = BVNModelMask(n_qubits_input=n_qubits, in_dim=X.shape[1], mask_resolution=1, interference=interference, standard_BV=standard_BV)
        if mode is None:
            dev = qml.device('lightning.qubit', wires=bv_model.ancilla_wires[-1] + 1)
            counts = bv_model.run(dev, state_vector, inputs, shots=100)
        if mode == 'noise':
            dev = qml.device("qiskit.aer", wires=bv_model.ancilla_wires[-1]+1, backend="aer_simulator", noise_model=noise_model)
            counts = bv_model.run(dev, state_vector, inputs, shots=100)
        if mode == 'random':
            dev = qml.device('lightning.qubit', wires=bv_model.ancilla_wires[-1] + 1)
            counts = bv_model.run(dev, state_vector, inputs, shots=100)
            if standard_BV:
                l_effective = bv_model._input_wires[-1] + 1
                counts = {"".join(random.choices(["0", "1"], k=l_effective) + ['0'] * (len(x) - l_effective)): 0 for x in counts}
            else:
                counts = {"".join(random.choices(["0", "1"], k=len(x))): 0 for x in counts}

        if n_qubits < 5:
            qml.draw_mpl(bv_model.circuit)(state_vector, inputs)
            plt.show()

        results = bv_model.get_function_from_counts(counts, X, y, gram_scalar=.1)
        title = f'{"Std." if standard_BV else "Gen."} BVN'

        # Results reconstruction
        y_pred = bv_model.predict(X, results)
        y_full_pred = bv_model.predict(X_full, results)
        plot(X, y_pred, X_full=X_full, y_full=y_full, y_full_pred=y_full_pred, accuracy=round_to_labels_acc(y_full_pred, y_full, y_full), vmin=y.min(), vmax=y.max(), title=title)

        # Visualise decision boundaries on full domain, real
        X_full_new = np.random.uniform(0, 2**n_qubits-1, (1024, X.shape[1]))
        y_pred_new = bv_model.predict(X_full_new, results)
        plot(X_full_new, y_pred_new, vmin=y.min(), vmax=y.max(), title=title)
