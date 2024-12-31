class Chromosome:
    """
    A simple data structure representing a genetic 'Chromosome'.
    
    Attributes:
        gene (np.ndarray): The feature vector for this chromosome.
        loss (float): The fitness or loss value associated with this chromosome.
    """
    def __init__(self, gene, loss):
        self.gene = gene
        self.loss = loss


class oversampling:
    """
    This class implements an oversampling technique using a genetic algorithm.
    
    Attributes:
        df (pd.DataFrame): Original DataFrame that includes features and labels.
        x (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.
        feature_num (int): Number of features (dimension of x).
        initialFitnessFunction: A model with 'fit' and 'predict_proba' for fitness measurement.
        alpha_ratio (float): Relative weight for ML model's confidence in the final fitness.
        beta_ratio (float): Relative weight for domain-based or distance-based loss.
        dataTypeDict (dict): Maps each feature name to 'binary', 'category', or 'continuous'.
        describeData (dict): Statistical info (mean, std, min, max) for continuous features.
        islands (dict): Optional structure that may hold cluster/region info for minors/majors.
        features_mask (None or list): Feature selection mask if partial features are used.
        beta (float): Parameter for controlling certain domain aspects (if needed).
        k (int): Number of nearest neighbors to consider.
        threshold (float): Value for threshold-based logic (if used).
        genetic_iteration (int): Number of genetic algorithm iterations for each sample.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self,
                 df,
                 x,
                 y,
                 feature_num,
                 fitness_ml_model,
                 alpha_ratio=0.8,
                 beta_ratio=0.2,
                 features_mask=None,
                 beta=0.5,
                 k=30,
                 threshold=1,
                 genetic_iteration=20,
                 seed=2021,
                 dataTypeDict=None,
                 describeData=None,
                 islands=None):
        self.df = df
        self.x = x
        self.y = y
        self.initialFitnessFunction = fitness_ml_model
        self.alpha_ratio = alpha_ratio
        self.beta_ratio = beta_ratio
        self.dataTypeDict = dataTypeDict
        self.describeData = describeData
        self.islands = islands
        self.feature_num = feature_num
        self.features_mask = features_mask
        self.beta = beta
        self.k = k
        self.threshold = threshold
        self.initial_mutation_rate = 0.9
        self.mutation_rate = 0.9
        self.genetic_iteration = genetic_iteration

        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        # Identify which label is minority/majority
        self.tar = 1
        self.ms = np.where(self.y == 1)[0]  # minority indices
        self.ml = np.where(self.y == 0)[0]  # majority indices

        if len(self.ml) < len(self.ms):
            # If the above assumption is reversed, swap them
            temp = self.ms
            self.ms = self.ml
            self.ml = temp
            self.tar = 0

        self.ms_count = len(self.ms)
        self.ml_count = len(self.ml)

        # For storing new (oversampled) data if needed
        self.x_f = np.array([])
        self.y_f = np.array([])

        # Placeholder references
        # (These might be set or trained later)
        self.knn = None

        # Train the initial fitness function
        self.train_initial_fitness_function()


    def ref_model(self):
        """
        Initialize and train the KNeighborsClassifier based on the current x and y.
        """
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.knn.fit(self.x, self.y)


    def sigmoid(self, x):
        """
        Calculate the sigmoid of x.

        Args:
            x (float or np.ndarray): Input value(s).

        Returns:
            float or np.ndarray: Sigmoid function result.
        """
        return 1 / (1 + np.exp(-x))
    
        
    def calculate_replication_ratio(self, minorities, desired_size):
        """
        Calculate how many synthetic samples each minority instance should generate
        based on the ratio of minority to majority neighbors.

        Args:
            minorities (np.ndarray): Indices of minority samples.
            desired_size (int): Desired total size for the minority class.

        Returns:
            np.ndarray: Array of replication counts for each minority sample.
        """
        minProbabilities = []
        for ms_index in minorities:
            ratio = self.find_min_maj_ratio(ms_index)
            minProbabilities.append(ratio)

        minProbabilities = np.array(minProbabilities)
        practical_size = desired_size - len(minorities)
        if practical_size < 0:
            raise AssertionError("The desired size must be greater than the size of the current minority samples.")

        # Normalize
        minProbabilitiesNormal = minProbabilities / np.sum(minProbabilities)
        repRatioList = minProbabilitiesNormal * practical_size
        repRatioList = np.abs(np.round(repRatioList))

        return repRatioList



    def train_initial_fitness_function(self):
        """
        Train the initialFitnessFunction model on the full DataFrame, assuming
        that the last column is the label.
        """
        self.initialFitnessFunction.fit(self.df[self.df.columns[:-1]], self.df[self.df.columns[-1]])
        

    def calculate_trapped_loss(self, trapped: int, synthesized: np.array):
        """
        Compute the loss for a 'trapped' minority sample, emphasizing the relationship
        to nearby minority neighbors vs. major neighbors.

        Args:
            trapped (int): Index of the trapped minority sample.
            synthesized (np.ndarray): The newly generated sample.

        Returns:
            float: A combined fitness value considering both the model’s prediction
                (weighted by alpha_ratio) and the domain-based 'trapped' penalty
                (weighted by beta_ratio).
        """
        Ti = np.array([], dtype=int)  # array for trapped neighbors
        Mi = np.array([], dtype=int)  # array for major neighbors around trapped neighbors
        neighbors = self.find_neighbors(trapped)
        neighbors = neighbors[neighbors != trapped]
        minor_neighbors = neighbors[np.in1d(neighbors, self.ms)]

        # We assume cfs_minor & its 'regions' and 'clusters' are assigned externally
        minority_trapped_index = np.where(self.ms == trapped)[0][0]
        cluster = self.cfs_minor.clusters[minority_trapped_index]

        # Identify neighbors in the same cluster that are also 'trapped'
        for i in minor_neighbors:
            minority_index = np.where(self.ms == i)[0][0]
            if minority_index in self.cfs_minor.regions['trapped']:
                if self.cfs_minor.clusters[minority_index] == cluster:
                    Ti = np.append(Ti, i)
                    neighbors_j = self.find_neighbors(i)
                    major_neighbors = neighbors_j[np.in1d(neighbors_j, self.ml)]
                    Mi = np.append(Mi, major_neighbors)

        # Calculate distances
        if len(Mi) != 0:
            Mi_maxima_distance = np.max(self.calculate_distance(synthesized, self.x[Mi]))
        else:
            Mi_maxima_distance = 0

        if len(Ti) != 0:
            Ti_maxima_distance = np.max(self.calculate_distance(synthesized, self.x[Ti]))
        else:
            Ti_maxima_distance = np.inf

        # Domain-based penalty
        loss_alpha = max(0, 1 - (np.nan_to_num(Ti_maxima_distance - Mi_maxima_distance)))

        if len(Ti) != 0:
            Ti_mean = np.mean(self.x[Ti], axis=0)
            cosine = np.absolute(
                np.dot(Ti_mean, synthesized) / (np.linalg.norm(Ti_mean) * np.linalg.norm(synthesized))
            )
        else:
            cosine = 0

        loss_beta = 1 / (1 - cosine) if (1 - cosine) != 0 else 1
        loss = 0.5 * (loss_alpha) + 0.5 * (loss_beta)
        loss = np.nan_to_num(loss)
        loss = 1 - self.sigmoid(loss)

        # Combine with ML model prediction
        model_prob = self.initialFitnessFunction.predict_proba(synthesized.reshape(1, -1))[:, self.tar][0]
        final_fitness = np.max((self.alpha_ratio * model_prob) + (self.beta_ratio * loss), 0)

        return final_fitness


    def calculate_inland_borderline_loss(self, in_bo: int, synthesized: np.array):
        """
        Compute the loss for 'inland' or 'borderline' minority samples, focusing on
        maximizing distance from major neighbors and adjusting distance to inland/border neighbors.

        Args:
            in_bo (int): Index of the inland/borderline minority sample.
            synthesized (np.ndarray): The newly generated sample.

        Returns:
            float: A combined fitness value that accounts for model prediction
                and domain-based distance metrics.
        """
        Ii = np.array([], dtype=int)  # inland neighbors
        Bi = np.array([], dtype=int)  # borderline neighbors
        Mi = np.array([], dtype=int)  # major neighbors

        neighbors = self.find_neighbors(in_bo)
        neighbors = neighbors[neighbors != in_bo]
        minor_neighbors = neighbors[np.in1d(neighbors, self.ms)]

        minority_in_bo_index = np.where(self.ms == in_bo)[0][0]
        cluster = self.cfs_minor.clusters[minority_in_bo_index]

        # Identify inland or borderline neighbors in the same cluster
        for i in minor_neighbors:
            minority_index = np.where(self.ms == i)[0][0]
            if self.cfs_minor.clusters[minority_index] == cluster:
                if minority_index in self.cfs_minor.regions['inland']:
                    Ii = np.append(Ii, i)
                    neighbors_j = self.find_neighbors(i)
                    major_neighbors = neighbors_j[np.in1d(neighbors_j, self.ml)]
                    Mi = np.append(Mi, major_neighbors)

                    minor_neighbors_j = neighbors_j[np.in1d(neighbors_j, self.ms)]
                    for j in minor_neighbors_j:
                        mn_index = np.where(self.ms == j)[0][0]
                        if mn_index in self.cfs_minor.regions['borderline']:
                            Bi = np.append(Bi, j)

        # Distances
        if len(Mi) != 0:
            major_max_distance = np.max(self.calculate_distance(synthesized, self.x[Mi]))
        else:
            major_max_distance = np.inf

        if len(Ii) != 0:
            inland_maxima_distance = np.max(self.calculate_distance(synthesized, self.x[Ii]))
        else:
            inland_maxima_distance = np.inf

        if len(Bi) != 0:
            margine = np.max(self.calculate_distance(synthesized, self.x[Bi]))
        else:
            # if there are no borderline neighbors, use a large margin
            margine = 2 * major_max_distance

        # Domain-based distance loss
        loss = 0.5 * (inland_maxima_distance ** 2
                    + max(0, np.nan_to_num(margine - major_max_distance)) ** 2)
        loss = 1 - self.sigmoid(loss)

        # Combine with ML model predictions
        model_prob = self.initialFitnessFunction.predict_proba(synthesized.reshape(1, -1))[:, self.tar][0]
        final_fitness = np.max((self.alpha_ratio * model_prob) + (self.beta_ratio * loss), 0)

        return final_fitness


    def fitness_function(self, _type):
        """
        Return the appropriate fitness calculation function based on minority region type.

        Args:
            _type (str): Either 'inland', 'borderline', or 'trapped'.

        Returns:
            function: A method reference that calculates the corresponding fitness.
        """
        if _type in ['inland', 'borderline']:
            return self.calculate_inland_borderline_loss
        elif _type == 'trapped':
            return self.calculate_trapped_loss


    def find_index(self, point, dataset):
        """
        Find the index in 'dataset' where 'point' is exactly located
        by summing the differences.

        Args:
            point (np.ndarray): The data point to find.
            dataset (np.ndarray): Array of points.

        Returns:
            int: The index of 'point' in 'dataset'.
        """
        differences = np.sum(dataset - point, axis=1)
        index = np.where(differences == 0)[0][0]
        return index


    def calculate_distance(self, point: np.array, dataset: np.array):

      return HEEM(point, dataset, self.features_mask)


    def calculate_distance(self, point: np.array, dataset: np.array):
        """
        Calculate the distance between a single 'point' and each row in 'dataset'.

        If self.features_mask is set (and at least one element is True, indicating
        categorical columns), use the HEEM metric. Otherwise, use standard Euclidean distance.

        Args:
            point (np.ndarray): Single data point (1D array).
            dataset (np.ndarray): Matrix of data points (2D array).

        Returns:
            np.ndarray: Array of distances with shape (len(dataset),).
        """
        # If no mask or mask is all False => purely numerical => Euclidean
        if self.features_mask is None or not np.any(self.features_mask):
            return np.linalg.norm(dataset - point, axis=1)
        else:
            # Use HEEM for combined categorical + numerical data
            return HEEM(point, dataset, self.features_mask)
    
        
    def calculate_total_distance(self, point: np.array) -> np.array:
        if self.features_mask is None or not np.any(self.features_mask):
            return np.linalg.norm(self.x - point, axis=1)
        else:
            return HEEM(point, self.x, self.features_mask)
        
        
    def find_neighbors(self, i):
        """
        Retrieve the indices of the k-nearest neighbors to x[i].

        Args:
            i (int): Index of the reference sample in self.x.

        Returns:
            np.ndarray: Indices of the k nearest neighbors.
        """
        distances = self.calculate_total_distance(self.x[i])
        # Sort by ascending distance, exclude the sample itself
        return np.argsort(distances)[1 : self.k + 1]



    def generate_synthetic_samples(self, desire=3000):
        """
        Main driver for oversampling:
        1) Calculate replication ratio for minority samples
        2) Cluster minority/major samples (build cfs_minor, cfs_major)
        3) For each minority sample, run genetic_algorithm to create new synthetic points.

        Args:
            desire (int): The target total number of minority samples.

        Returns:
            (np.ndarray, np.ndarray): The updated feature matrix (self.x) and label vector (self.y).
        """
        # 1. Calculate the replication ratio for each minority sample
        replication_ratio = self.calculate_replication_ratio(self.ms, desire)

        # Sort minority indices by descending replication ratio
        minor_sorting = np.argsort(replication_ratio)[::-1]
        self.ms = self.ms[minor_sorting]

        # 2. Cluster minority & major sets (assumes cluster_coordinates is available)
        self.cfs_minor = cluster_coordinates(self.x[self.ms], features_mask=self.features_mask, neighbors_num=self.k)
        self.cfs_major = cluster_coordinates(self.x[self.ml], features_mask=self.features_mask, neighbors_num=self.k)

        # 3. Generate synthetic samples using the genetic algorithm
        for index, i in tqdm(enumerate(self.ms), desc="Processing"):
            if len(self.y[self.y == 1]) >= desire:
                break

            # Determine region: 'inland', 'borderline', or 'trapped'
            if index in self.cfs_minor.regions['inland']:
                region = 'inland'
            elif index in self.cfs_minor.regions['borderline']:
                region = 'borderline'
            else:
                region = 'trapped'

            ancesstor = i
            h = replication_ratio[index]

            # Generate new samples h times
            for _ in range(int(h)):
                neighbors = self.find_neighbors(i)
                # Add the original sample into the neighbor pool
                neighbors = np.append(neighbors, i)

                new_sample = self.genetic_algorithm(
                    ancesstor=ancesstor,
                    fit_function=self.fitness_function(region),
                    neighbors=neighbors,
                    iterations=self.genetic_iteration,
                    size=self.feature_num,
                    region=region
                )

                # Check if the newly generated sample belongs to minority label
                temp_y = 1
                if temp_y == self.tar:
                    # Append to self.x, self.y
                    self.x = np.append(self.x, new_sample.reshape(1, -1), axis=0)
                    self.y = np.append(self.y, temp_y)

        return self.x, self.y


    def choose_random_samples(self, minority_set, size, prob):
        """
        Randomly choose 'size' samples from minority_set according to probability 'prob'.

        Args:
            minority_set (list or np.ndarray): Indices of minority samples.
            size (int): Number of samples to choose.
            prob (np.ndarray): Probability distribution over minority_set.

        Returns:
            list: List of chosen indices.
        """
        indices = [np.random.choice(minority_set, p=prob) for _ in range(size)]
        return indices


    def find_major(self, i):
        """
        Count how many neighbors of sample i are from the majority class.

        Args:
            i (int): Index in self.x.

        Returns:
            int: Number of major class neighbors.
        """
        neighbors = self.find_neighbors(i)
        majority_count = sum(1 for j in neighbors if j in self.ml)
        return majority_count


    def find_avg_distance(self, i):
        """
        Compute the average distance between sample i and its minority neighbors.

        Args:
            i (int): Index in self.x.

        Returns:
            float: The average Euclidean distance to minority neighbors.
        """
        neighbors = self.find_neighbors(i)
        dis = 0
        count = 0
        for j in neighbors:
            if j in self.ms:
                dis += distance.euclidean(self.x[i], self.x[j])
                count += 1

        if count == 0:
            return 0
        return dis / count


    def find_min_maj_ratio(self, i):
        """
        Calculate the ratio of minority neighbors to majority neighbors for sample i.

        Args:
            i (int): Index in self.x.

        Returns:
            float: Ratio = (# of minority neighbors) / (# of majority neighbors).
                If the denominator (major neighbors) is zero, returns 1.
        """
        neighbors = self.find_neighbors(i)
        count_min = sum(1 for j in neighbors if j in self.ms)
        count_maj = self.k - count_min  # total neighbors is k
        if count_maj == 0:
            return 1
        else:
            return count_min / count_maj

    def calculate_probabilities(self, x):
        """
        Convert a set of values into a probability distribution using a softmax approach.

        Args:
            x (np.ndarray): Array of values (e.g., losses or fitness scores).

        Returns:
            np.ndarray: Corresponding probabilities summing to 1.
        """
        # Shift values by subtracting the max to improve numeric stability
        x -= np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)

        # If numerical instability leads to NaN, fall back to uniform distribution
        if np.any(np.isnan(softmax_x)):
            return np.full_like(x, 1.0 / x.size)

        return softmax_x
