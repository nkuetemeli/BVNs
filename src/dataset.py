import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.datasets import make_moons, make_circles, make_blobs, load_iris
from sklearn.preprocessing import MinMaxScaler

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import io

import torch
import torchvision.transforms as T

import numpy as np
from scipy.signal import resample
from collections import defaultdict, Counter

import numpy as np
import itertools

from collections import defaultdict, Counter


# ================================================================
#  UTILITY FUNCTIONS
# ================================================================
def convert_to_grid(X, scale):
    """Scale continuous features into integer grid coordinates."""
    scaler = MinMaxScaler(feature_range=(0, scale))
    X_scaled = scaler.fit_transform(X)
    return np.round(X_scaled).astype(int)


def fill_hyper_grid(X, y, scale, dummy_fill=None):
    if dummy_fill is None or dummy_fill[0] == 0 or not 0 <= dummy_fill[1] <= 1:
        return X, y

    dummy_value, percentage = dummy_fill

    grid_axis = [int(i) for i in range(int(scale) + 1)]
    all_possible_points = np.array(list(itertools.product(grid_axis, repeat=X.shape[1])), dtype=np.int64)
    set_all = {tuple(int(v) for v in p) for p in all_possible_points.tolist()}
    set_existing = {tuple(int(v) for v in p) for p in X.tolist()}
    set_remaining = set_all - set_existing
    remaining_points = np.array(list(set_remaining))

    num_to_sample = int(np.ceil(len(remaining_points) * percentage))
    sampled_indices = np.random.choice(len(remaining_points), size=num_to_sample, replace=False)
    dummy_points = remaining_points[sampled_indices]
    dummy_labels = np.full(len(dummy_points), dummy_value, dtype=y.dtype)

    X_final = np.vstack((X, dummy_points))
    y_final = np.concatenate((y, dummy_labels))

    return X_final, y_final



def clean_dataset(X, y, strategy="majority", verbose=True):
    """
    Clean dataset by resolving conflicts where identical rows in X have different labels in y.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (N, k).
    y : np.ndarray
        Labels of shape (N,).
    strategy : str, optional
        How to resolve conflicts:
        - "majority": keep one row with the majority label
        - "remove": drop all conflicting rows
        - "first": keep the first occurrence
    verbose : bool, optional
        If True, prints the number of removed samples.

    Returns
    -------
    X_clean, y_clean : np.ndarray
        Cleaned dataset.
    """
    grouped = defaultdict(list)
    for idx, row in enumerate(X):
        grouped[tuple(row)].append(idx)

    keep_indices = []
    removed_count = 0

    for row, indices in grouped.items():
        labels = [y[i] for i in indices]
        if len(set(labels)) == 1:
            # no conflict → keep all
            keep_indices.extend(indices)
        else:
            # conflict → resolve based on strategy
            if strategy == "majority":
                most_common = Counter(labels).most_common(1)[0][0]
                # keep one representative with majority label
                keep_indices.append(indices[0])
                y[indices[0]] = most_common
                removed_count += len(indices) - 1
            elif strategy == "remove":
                # drop all conflicting rows
                removed_count += len(indices)
            elif strategy == "first":
                # keep the first occurrence
                keep_indices.append(indices[0])
                removed_count += len(indices) - 1

    if verbose:
        print(f"Removed {removed_count} samples due to conflicts.")

    return X[keep_indices], y[keep_indices]


def plot(X, y, X_full=None, y_full=None, y_full_pred=None, accuracy=None, vmin=-1, vmax=1, title=''):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    y = np.array(y.tolist())
    y_full_pred = np.array(y_full_pred.tolist()) if y_full_pred is not None else None

    if X.ndim == 2 and X.shape[1] == 1:
        if X_full is not None:
            plt.plot(X_full, y_full, label="target")
            plt.plot(X_full, y_full_pred, label="fitted")
        plt.scatter(X, y, label="fitted")
        plt.title(title)
        plt.legend()
        plt.show()
    if X.ndim == 2 and X.shape[1] == 2:
        cmap = "RdBu_r"

        cbar = axes.scatter(*X.T, c=y, label="fitted", cmap=cmap)
        axes.set_title(title)

        axes.set_title(f"accuracy = {round(accuracy, 2) if accuracy is not None else 'None'}, "+title)
        axes.grid(True, linestyle="--", linewidth=0.5)
        axes.legend()
        axes.axis('image')

        cax = fig.add_axes([.91, 0.11, 0.02, 0.77])
        fig.colorbar(cbar, cax=cax)
        plt.show()
    else:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]
        cmap = LinearSegmentedColormap.from_list("custom3", colors, N=256)

        cbar = axes.scatter(X[:, 0], X[:, 1], c=y, marker='o', label="training", vmin=vmin, vmax=vmax, cmap=cmap)
        if X_full is not None:
            complement_indices = np.array([i for i, row in enumerate(X_full) if tuple(row) not in set(map(tuple, X))])
            if len(complement_indices) > 0:
                axes.scatter(X_full[complement_indices, 0], X_full[complement_indices, 1], c=y_full_pred[complement_indices], marker='s', label='test', vmin=vmin, vmax=vmax, cmap=cmap)

        axes.set_title(f"accuracy = {round(accuracy, 2) if accuracy is not None else 'None'}, "+title)
        axes.grid(True, linestyle="--", linewidth=0.5)
        axes.legend()
        axes.axis('image')

        cax = fig.add_axes([.91, 0.11, 0.02, 0.77])
        fig.colorbar(cbar, cax=cax)
        plt.show()



def round_to_labels_acc(y_pred, y, labels):
    y = np.array(y.tolist()).flatten()
    y_pred = np.array(y_pred.tolist()).flatten()
    labels = np.array(labels).flatten()
    snapped = np.array([min(labels, key=lambda x: abs(x - float(val))) for val in y_pred])
    acc = (y == snapped).mean()
    return acc


# ================================================================
#  SYNTHETIC BINARY DATASETS (your original)
# ================================================================


def shape_dataset_2d(shape="circle", scale=15, promise='Sigmoid', random_seed=None):
    """Synthetic 2D binary datasets on a full integer grid."""

    grid_points = np.array([(i, j) for i in range(scale + 1) for j in range(scale + 1)])
    labels = np.zeros(len(grid_points), dtype=int)

    center = scale / 2
    n_samples = (scale + 1) ** 2

    if shape == "moons":
        # Generate moons dataset and scale to fit the grid
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=random_seed)
        X = convert_to_grid(X, scale)  # Rescale to grid
        unique_points, indices = np.unique(X, axis=0, return_index=True)  # Remove duplicates
        grid_points, labels = unique_points, y[indices]  # Return unique points and corresponding labels

    elif shape == "blobs":
        # Generate blobs dataset and scale to fit the grid
        X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.5, random_state=random_seed)
        X = convert_to_grid(X, scale)  # Rescale to grid
        unique_points, indices = np.unique(X, axis=0, return_index=True)
        grid_points, labels = unique_points, y[indices]

    elif shape == "sk_circles":
        # Generate circles dataset and scale to fit the grid
        X, y = make_circles(n_samples=n_samples, noise=0.1, random_state=random_seed, factor=0.5)
        X = convert_to_grid(X, scale)  # Rescale to grid
        unique_points, indices = np.unique(X, axis=0, return_index=True)
        grid_points, labels = unique_points, y[indices]

    elif shape == "circle":
        # Define a circle at the center of the grid
        radius = (scale / 3) ** 2
        for idx, (x, y) in enumerate(grid_points):
            distance = (x - center) ** 2 + (y - center) ** 2
            labels[idx] = 1 if distance <= radius else 0

    elif shape == "cross":
        # Define a cross pattern
        for idx, (x, y) in enumerate(grid_points):
            if (x > scale // 3 and x < 2 * scale // 3) or (y > scale // 3 and y < 2 * scale // 3):
                labels[idx] = 1
            else:
                labels[idx] = 0

    elif shape == "checkerboard":
        # Define a checkerboard pattern
        for idx, (x, y) in enumerate(grid_points):
            labels[idx] = 1 - (x + y) % 2  # Alternate 0 and 1

    elif shape == "diamond":
        for idx, (x, y) in enumerate(grid_points):
            if abs(x - center) + abs(y - center) <= scale / 2:
                labels[idx] = 1

    elif shape == "h_stripes":
        for idx, (x, y) in enumerate(grid_points):
            labels[idx] = x % 2

    elif shape == "v_stripes":
        for idx, (x, y) in enumerate(grid_points):
            labels[idx] = (y // 8) % 2

    elif shape == "spiral":
        for idx, (x, y) in enumerate(grid_points):
            distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            angle = np.arctan2(y - center, x - center) + np.pi  # Angle in radians
            spiral_band = int((distance + angle / np.pi * 3) // (scale / 8)) % 2
            labels[idx] = spiral_band

    elif shape == "permutation":
        labels = np.eye(scale + 1)
        np.random.shuffle(labels)
        labels = labels.flatten()

    elif shape == "relu":
        for idx, (x, y) in enumerate(grid_points):
            if (scale - x) < y:
                labels[idx] = 1
    else:
        labels = np.array(list(np.random.choice(2, len(grid_points), replace=True)))
        print("Invalid shape. Returning a random one.")

    if shape != 'promise':
        labels = labels*2-1
    else:
        labels = labels + 1

    return grid_points, labels


# ================================================================
#  REAL-WORLD MULTI-CLASS DATASETS
# ================================================================

def load_iris_grid(scale=15):
    iris = load_iris()
    X_grid = convert_to_grid(iris.data, scale)
    return X_grid, iris.target + 1


def load_penguins_grid(scale=15):
    import seaborn as sns
    penguins = sns.load_dataset("penguins").dropna()

    features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    X = penguins[features].values
    species_map = {species: i for i, species in enumerate(penguins["species"].unique())}
    y = penguins["species"].map(species_map).values + 1

    X_grid = convert_to_grid(X, scale)
    return X_grid, y


# ================================================================
#  1d regression data
# ================================================================
def generate_1d_function(scale, frequency, seed=0, n_samples=200):
    """
    Generate the target function using a random vector of given frequency.
    Returns:
        X, y, X_target, y_target, rand_vec
    """
    seed = seed

    np.random.seed(seed)
    rand_vec = np.random.randn(frequency)

    np.random.seed(seed*frequency if seed is not None else None)
    num_changes = min(3, frequency)  # choose up to 3 positions
    indices = np.random.choice(frequency, num_changes, replace=False)
    rand_vec[indices] += np.random.uniform(-10, 10, size=len(indices))
    np.random.seed(seed)

    X_target = np.linspace(0, scale, n_samples)[:, None]
    y_target = resample(rand_vec, len(X_target))

    X = np.arange(scale+1)[:, None]
    y = np.interp(X.squeeze(), X_target.squeeze(), y_target).flatten()

    return X_target, y_target, X, y, rand_vec

# ================================================================
#  UNIFIED DATASET LOADER
# ================================================================

def load_dataset(name, scale=31, promise='Sigmoid', percentage=100, frequency=10, random_seed=None):
    """Unified entry point for all datasets."""
    name = name.lower()

    X_target, y_target, rand_vec = None, None, None
    # Real-world datasets
    if name == "iris":
        X_full, y_full = load_iris_grid(scale)
    elif name == "penguins":
        X_full, y_full = load_penguins_grid(scale)
    elif name == "1d":
        X_full, y_full, X, y, rand_vec = generate_1d_function(scale, frequency=frequency, seed=random_seed, n_samples=200)
    else:
        X_full, y_full = shape_dataset_2d(shape=name, scale=scale, promise=promise, random_seed=random_seed)

    if name != "1d":
        if name == "promise":
            np.random.seed(0)
        indices = np.random.choice(len(X_full), size=int(len(X_full) * percentage / 100), replace=False)
        np.random.seed(None)
        X = X_full[indices]
        y = y_full[indices]

    X, y = clean_dataset(X, y)
    return X, y, X_full, y_full, rand_vec


def load_image(scale, bv_model=False):
    # img = io.imread('../data/qml_image_flower.jpg')
    img = io.imread('../data/qml_image_dog.jpg')
    img = rgb2gray(img)
    img_coarse = resize(img, (scale, scale), anti_aliasing=True)  # Resize image
    img_fine = resize(img, (2 * scale, 2 * scale), anti_aliasing=True)  # Resize image

    transform_coarse = T.Compose([T.ToTensor()])
    transform_fine = T.Compose([T.ToTensor()])

    img_tensor_coarse = transform_coarse(img_coarse).permute(1, 2, 0)
    img_tensor_fine = transform_fine(img_fine).permute(1, 2, 0)


    # Training data: 32x32 coordinates and colors
    H_coarse, W_coarse, C = img_tensor_coarse.shape
    if bv_model:
        coords_coarse = np.stack(np.meshgrid(np.arange(0, W_coarse), np.arange(0, H_coarse)), axis=-1).reshape(-1, 2)
        coords_coarse = coords_coarse.astype(np.int32)
        colors_coarse = img_coarse.reshape(-1, 1)
    else:
        coords_coarse = np.stack(np.meshgrid(np.linspace(0, 1, W_coarse), np.linspace(0, 1, H_coarse)), axis=-1).reshape(-1, 2)
        coords_coarse = torch.tensor(coords_coarse, dtype=torch.float32).unsqueeze(0)
        colors_coarse = img_coarse.reshape(-1, 1)
        colors_coarse = torch.tensor(colors_coarse, dtype=torch.float32).unsqueeze(0)

    # Test data: 64x64 coordinates for generation
    H_fine, W_fine, C = img_tensor_fine.shape
    if bv_model:
        coords_fine = np.stack(np.meshgrid(np.linspace(0, W_coarse, W_fine), np.linspace(0, H_coarse, H_fine)), axis=-1).reshape(-1, 2)
        coords_fine = coords_fine.astype(np.int32)
        colors_fine = img_fine.reshape(-1, 1)
    else:
        coords_fine = np.stack(np.meshgrid(np.linspace(0, 1, W_fine), np.linspace(0, 1, H_fine)), axis=-1).reshape(-1, 2)
        coords_fine = torch.tensor(coords_fine, dtype=torch.float32).unsqueeze(0)
        colors_fine = img_fine.reshape(-1, 1)
        colors_fine = torch.tensor(colors_fine, dtype=torch.float32).unsqueeze(0)
    return img_coarse, img_fine, (coords_coarse, colors_coarse), (colors_fine, coords_fine)


# ================================================================
#  DEMO
# ================================================================
if __name__ == "__main__":
    # Generate and visualize datasets
    shapes = {0: 'promise', 1: 'moons', 2: 'blobs', 3: 'sk_circles', 4: 'circle', 5: 'cross', 6: 'checkerboard',
              7: 'diamond', 8: 'h_stripes', 9: 'v_stripes', 10: 'spiral', 11: 'permutation', 12: 'relu', 13: 'random',
              20: 'penguins',
              21: 'iris',
              22: '1d'}
    shape = shapes[21]

    X, y, X_target, y_target, rand_vec = load_dataset(shape, scale=15)
    plot(X, y, X_full=X_target, y_full=y_target, y_full_pred=y_target, vmin=y.min(), vmax=y.max())

