import time
from bvn_rect_act import *
from bvn_relu_act import *



if __name__ == '__main__':
    n_qubits = 6
    scale = 2 ** n_qubits

    img_coarse, img_fine, (coords_coarse, colors_coarse), (colors_fine, coords_fine) = load_image(scale, bv_model=True)
    X, X_full = coords_coarse, coords_fine
    y, y_full = colors_coarse.flatten(), colors_fine.flatten()

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(6, 3)
    ax[0].imshow(img_coarse, cmap='bone')
    ax[0].set_title("Coarse", fontsize=14)
    ax[0].axis('off')
    ax[1].imshow(img_fine, cmap='bone')
    ax[1].set_title("Fine", fontsize=14)
    ax[1].axis('off')
    plt.show()

    x_list = [''.join([np.binary_repr(a, width=n_qubits) for a in x]) for x in X]

    amplitude_encoding = True
    state_vector = uniform_state(len(x_list[0]), x_list, y=y)

    interference = {0: 'Hadamard', 1: 'Fourier', 2: 'Chebyshev',}
    activation = {0: 'ReLU', 1: 'Modulo', 2: 'Sigmoid'}

    image_records = []
    for standard_BV in [True, False]:

        tic = time.time()

        bv_model = BVNModelMask(n_qubits_input=n_qubits, in_dim=X.shape[1], mask_resolution=5, interference=interference[2], standard_BV=standard_BV)
        dev = qml.device('lightning.qubit', wires=bv_model.ancilla_wires[-1]+1)
        counts = bv_model.run(dev, state_vector, inputs=[], shots=10000)

        # if n_qubits < 5:
        #     qml.draw_mpl(bv_model.circuit)(state_vector, inputs=inputs)
        #     plt.show()

        results = bv_model.get_function_from_counts(counts, X, y, gram_scalar=.1)
        title = f'{"Std." if standard_BV else "Gen."} BVN'

        toc = time.time() - tic

        # # Results reconstruction
        y_pred = bv_model.predict(X, results)
        y_full_pred = bv_model.predict(X_full, results)

        # === 6. Generate image and compare with target ===
        print(f"Generating {2 * scale}x{2 * scale} image from trained model...")
        pred_img_coarse = y_pred.reshape(img_coarse.shape)
        pred_img_fine = y_full_pred.reshape(img_fine.shape)

        # Save results in dictionary
        record = {
            "standard_BV": standard_BV,
            "y": img_coarse,
            "y_full": img_fine,
            "y_pred": pred_img_coarse,
            "y_full_pred": pred_img_fine,
            "run_time": toc,
            "num_params": len(counts),
            'model': 'BV'
        }
        image_records.append(record)

        fig = plt.figure()
        fig.set_size_inches(12, 3)
        gs = fig.add_gridspec(1, 4, height_ratios=[1, ], width_ratios=[1, 1, 1, 1])

        ax = fig.add_subplot(gs[0, 0:2])
        ax.imshow(np.hstack([img_coarse, np.clip(pred_img_coarse, 0, 1)]), cmap='bone')
        ax.set_title("Coarse: True vs Reconstructed", fontsize=14)
        ax.axis('off')

        ax = fig.add_subplot(gs[0, 2:])
        ax.imshow(np.hstack([img_fine, np.clip(pred_img_fine, 0, 1)]), cmap='bone')
        ax.set_title("Fine: True vs Reconstructed", fontsize=14)
        ax.axis('off')


        title = f'{"Std." if standard_BV else "Gen."} BVN'
        fig.suptitle(title, fontsize=14)
        plt.show()
