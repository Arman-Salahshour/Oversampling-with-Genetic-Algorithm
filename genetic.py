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
