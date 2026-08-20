import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import pennylane as qml
import numpy as np

import pandas as pd


class CheapMask(qml.operation.Operation):
    num_wires = None
    grad_method = None  # non-differentiable

    def __init__(self, x_wires, mask_wires, superposition_wire, sigma_wire):
        flat_x = [w for reg in x_wires for w in reg]
        flat_y = [w for reg in mask_wires for w in reg]

        all_wires = qml.wires.Wires(flat_x + flat_y + superposition_wire + sigma_wire)

        self._hyperparameters = {
            'x_wires': x_wires,
            'mask_wires': mask_wires,
            'superposition_wire': superposition_wire,
            'sigma_wire': sigma_wire,
        }
        super().__init__(wires=all_wires)

    @staticmethod
    def compute_decomposition(wires, x_wires, mask_wires, superposition_wire, sigma_wire):
        op_list = []
        assert len(x_wires[0]) > len(mask_wires[0]), f'For super cells, the lengh† of x wires needto be greather than that of y wires.'
        super_size = len(x_wires[0]) - len(mask_wires[0])

        # circular schift
        control_reg = []
        for x_reg, y_reg in zip(x_wires, mask_wires):
            for i, y in enumerate(y_reg[::-1]):
                op_list.append(
                    qml.ctrl(
                        qml.Adder(k=2 ** (i + super_size), x_wires=x_reg),
                        control=y
                    )
                )
        for x_reg in x_wires:
            op_list.append(
                qml.adjoint(
                    qml.ctrl(
                        qml.Adder(k=2**super_size // 2, x_wires=x_reg),
                        control=superposition_wire
                    )
                )
            )

            control_reg.extend(x_reg[:-super_size])
        op_list.append(qml.ctrl(qml.X(wires=sigma_wire), control=control_reg, control_values=[0]*len(control_reg)))

        for x_reg in x_wires:
            op_list.append(
                qml.ctrl(
                    qml.Adder(k=2**super_size // 2, x_wires=x_reg),
                    control=superposition_wire
                )
            )
        for x_reg, y_reg in zip(x_wires, mask_wires):
            for i, y in enumerate(y_reg[::-1]):
                op_list.append(
                    qml.adjoint(
                        qml.ctrl(
                            qml.Adder(k=2 ** (i + super_size), x_wires=x_reg),
                            control=y
                        )
                    )
                )

        # Return the list of operations
        return op_list


def to_int(bits):
    return int(bits, 2)


def decode(bitstring, d, b_x, b_y, b_superposition, b_sigma):
    idx = 0
    X = []
    Y = []

    # x registers
    for i in range(d):
        xi = to_int(bitstring[idx:idx + b_x])
        idx += b_x
        X.append(xi)

    # y registers
    for i in range(d):
        yi = to_int(bitstring[idx:idx + b_y])
        idx += b_y
        Y.append(yi)

    # superposition register (shared)
    s = to_int(bitstring[idx:idx + b_superposition])
    idx += b_superposition

    # sigma register (shared)
    S = to_int(bitstring[idx:idx + b_sigma])
    idx += b_sigma

    return X, Y, s, S


if __name__ == "__main__":
    # ---- Configuration ----
    d = 2        # number of dimensions
    b_x = 6      # bits per x_i
    b_y = 3      # bits per y_i
    b_superposition = 1  # bits for shared sigma
    b_sigma = 1  # bits for shared sigma
    shots = 2000000

    total_x = d * b_x
    total_y = d * b_y
    total_wires = total_x + total_y + b_superposition + b_sigma

    dev = qml.device("default.qubit", wires=total_wires)

    # assign sub-registers
    x_wires = [list(range(i*b_x, (i+1)*b_x)) for i in range(d)]
    mask_wires = [list(range(total_x + i*b_y, total_x + (i+1)*b_y)) for i in range(d)]
    superposition_wire = list(range(total_x + total_y, total_x + total_y + b_superposition))
    sigma_wire = list(range(total_x + total_y + b_superposition, total_x + total_y + b_superposition + b_sigma))


    @qml.qnode(dev)
    def circuit():
        # superpose all inputs
        for wires in x_wires + mask_wires + [superposition_wire]:
            for w in wires:
                qml.Hadamard(w)

        # accumulate each dimension into the same sigma register
        CheapMask(
            x_wires=x_wires,
            mask_wires=mask_wires,
            superposition_wire=superposition_wire,
            sigma_wire=sigma_wire,
        )
        return qml.counts()
    circuit = qml.set_shots(circuit, shots=shots)



    # draw circuit
    qml.draw_mpl(circuit)()
    plt.show()

    # run circuit
    counts = circuit()





    # ---- Convert counts → expanded DataFrame of all measurement results ----
    records = []

    for bitstring, freq in counts.items():
        # Decode values (may need bitstring[::-1] depending on endian!)
        X, Y, s, S = decode(bitstring.replace(" ", ""), d, b_x, b_y, b_superposition, b_sigma)

        # store one row per shot
        records.append({
            "bitstring": bitstring,
            "X": tuple(X),
            "Y": tuple(Y),
            "s": s,
            "S": S,
        })

    df = pd.DataFrame(records)
    pivot = df.pivot_table(values="S", index="X", columns=["Y", "s"], aggfunc="sum", fill_value=0)

    # Pivot now contains columns (Y,s)
    pivot = df.pivot_table(
        values="S", index="X", columns=["Y", "s"], aggfunc="sum", fill_value=0
    )

    # Extract list of available Y values
    Y_vals = sorted(set(y for (y, s) in pivot.columns))

    ncols = 2 ** b_y
    nrows = int(np.ceil(len(Y_vals) / ncols))

    # Define your levels and colors
    levels = [-0.1, 0.5, 1.1]
    colors = ["lightgray", "steelblue"]
    # Build a colormap and a normalization
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    for s in [0, 1]:
        pivot_s = pivot.xs(s, level="s", axis=1)  # <-- selects only entries for this s

        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 18))
        axes = np.atleast_2d(axes)

        for idx, Y in enumerate(Y_vals):
            ax = axes[idx // ncols, idx % ncols]
            S_vals = pivot_s[Y].values

            shape = [2 ** b_x for _ in range(d)] + ([1] if d==1 else [])
            ax.imshow(S_vals.reshape(shape), origin='lower', cmap=cmap, norm=norm)
            ax.set_title(f"t={Y}, s={s}", fontsize=18)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"s register value = {s}", fontsize=40)
        plt.tight_layout()
        plt.show()
