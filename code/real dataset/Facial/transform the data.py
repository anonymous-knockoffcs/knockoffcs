import numpy as np
import pickle

variables = np.load("Original Dataset_variables.npy", allow_pickle=True).T
print(variables.shape)
labels = np.load("Original Dataset_labels.npy", allow_pickle=True)
row_variances = np.var(variables, axis=1, ddof=1)
row_std = np.sqrt(row_variances)
std = 0.5 * row_std.mean()
noise_dimensions = 475 
low_dim = 100

def add_noise_and_shuffle(vectors, noise_dims, noise_scale=0.1):
    """
    Add Gaussian noise dimensions to each sample and randomly permute all dimensions.

    Parameters
    ----------
    vectors : ndarray
        Input matrix of shape (n_dims, n_samples).
    noise_dims : int
        Number of Gaussian noise dimensions to append.
    noise_scale : float, optional
        Standard deviation of the Gaussian noise.

    Returns
    -------
    shuffled_vectors : ndarray
        The augmented matrix after adding noise dimensions and randomly
        permuting all dimensions.
    shuffle_idx : ndarray
        Permutation indices used for shuffling the dimensions. These can be
        used to recover the positions of the original signal dimensions.
    """
    n_dims, n_samples = vectors.shape

    # Generate Gaussian noise
    noise = np.random.normal(0, noise_scale, (noise_dims, n_samples))

    # Concatenate the original signal with the noise dimensions
    extended_vectors = np.vstack([vectors, noise])

    # Generate a random permutation of all dimensions
    shuffle_idx = np.random.permutation(n_dims + noise_dims)

    # Permute the dimensions according to the generated indices
    shuffled_vectors = extended_vectors[shuffle_idx, :]

    return shuffled_vectors, shuffle_idx

extended_variables, dimension_indices = add_noise_and_shuffle(variables, noise_dimensions, std)
print(extended_variables.shape)
high_dim = extended_variables.shape[0]
A = np.random.randn(low_dim, high_dim)

with open("real measurement matrix.pkl", "wb") as f:
    pickle.dump(A, f)

A_fake = np.random.randn(low_dim, high_dim)
with open("Another measurement matrix.pkl", "wb") as f:
    pickle.dump(A_fake, f)

transformed_variables = A @ extended_variables
print(transformed_variables.shape)

np.save("Processed_vectors.npy", extended_variables, allow_pickle=True)
np.save("Dimension_indices.npy", dimension_indices, allow_pickle=True)
np.save('transformed_variables.npy', transformed_variables, allow_pickle=True)
