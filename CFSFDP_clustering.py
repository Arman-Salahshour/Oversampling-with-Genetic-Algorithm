from constants import *


@jit
def HEEM(reference_record, coordinates, mask):
    """
    Computes hybrid entropy-enhanced Euclidean metric (HEEM) for clustering.

    Args:
        reference_record (np.array): The reference data point.
        coordinates (np.array): Dataset containing all data points.
        mask (np.array): Boolean array indicating categorical features (True) and numerical features (False).

    Returns:
        np.array: Distances between the reference record and all data points.
    """
    # Handle categorical features if present
    if mask.any():
        dataset_cat = coordinates[:, mask]  # Extract categorical columns
        ref_cat = reference_record[mask]  # Extract categorical reference values
        Truths = (ref_cat == dataset_cat)  # Compare reference with dataset
        p_distribution = Truths.mean(axis=0)  # Proportion of matching values
        epsilon = 1e-12  # Prevent log(0)
        entropies = -0.5 * (p_distribution * np.log(p_distribution + epsilon) +
                            (1 - p_distribution) * np.log(1 - p_distribution + epsilon))
        entropies_weights = (entropies.max() - entropies) / entropies.sum()  # Weighting by entropy
        categorical_distances = ~Truths * entropies_weights  # Weighted distances for categorical features

    # Handle numerical features
    dataset_num = coordinates[:, ~mask]  # Extract numerical columns
    ref_num = reference_record[~mask]  # Extract numerical reference values
    std_deviation = np.std(dataset_num, axis=0)  # Standard deviation for scaling
    std_deviation[std_deviation == 0] = 1  # Avoid division by zero
    numerical_distances = np.abs(ref_num - dataset_num) / (4 * std_deviation)  # Scaled distances

    # Combine distances if categorical features exist
    if mask.any():
        all_distances = np.concatenate((categorical_distances, numerical_distances), axis=1)
        return np.linalg.norm(all_distances, ord=2, axis=1)  # Euclidean norm
    else:
        return np.linalg.norm(numerical_distances, ord=2, axis=1)



def calculate_importance(density, sigma, sigma_weight=0.4):
    """
    Calculates importance scores based on density and sigma values.

    Args:
        density (np.array): Density values for data points.
        sigma (np.array): Sigma values for data points.
        sigma_weight (float): Weight assigned to sigma in the importance calculation. Default is 0.4.

    Returns:
        np.array: Importance scores for data points.
    """
    # Normalize density and sigma
    normalized_density = (density - np.min(density)) / (np.max(density) - np.min(density))
    normalized_sigma = (sigma - np.min(sigma)) / (np.max(sigma) - np.min(sigma))

    # Compute weighted importance scores
    importance = (sigma_weight * normalized_sigma) + ((1 - sigma_weight) * normalized_density)

    return importance



class CFSFDP:
    def __init__(self, coordinates: list, k: int, distance_type: int = 2, features_mask=None):
        """
        Initializes the CFSFDP clustering algorithm.

        Args:
            coordinates (list): Array of data points.
            k (int): Number of neighbors for density estimation.
            distance_type (int): Distance metric type (1: Manhattan, 2: Euclidean, np.inf: Chebyshev).
            features_mask (np.array, optional): Boolean array indicating categorical and numerical features.
        """
        self.coordinates = coordinates
        self.k = k
        self.distance_type = distance_type
        self.features_mask = features_mask


    def calculate_density(self, quantile=0.02) -> np.array:
        """
        Computes density for each point based on its neighbors.

        Args:
            quantile (float): Proportion of neighbors to consider for density estimation.

        Returns:
            np.array: Density values for all data points.
        """
        density = []
        for point in self.coordinates:
            distances = self.calculate_distance(point)
            dc = np.quantile(distances, quantile)  # Cutoff distance
            dij = np.exp(-1 * (self.calculate_distance(point) / dc) ** 2)  # Gaussian kernel
            density.append(np.sum(dij))  # Summing over neighbors
        return np.array(density)
    
    
