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
    
    
