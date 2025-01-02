# GeneticOversampler
A Python implementation that combines genetic algorithms and clustering techniques to address class imbalance in datasets through oversampling.

## Overview
`GeneticOversampler` uses a hybrid approach, integrating genetic algorithms (GA) and clustering methods, to generate synthetic samples for minority classes in imbalanced datasets. This innovative combination enhances the diversity and relevance of synthetic samples, improving downstream machine-learning model performance.

## Key Algorithms

### Genetic Algorithm (GA)
The genetic algorithm is central to generating synthetic samples. It mimics natural selection through:

1. **Initialization**: Minority class samples are selected as initial chromosomes.
2. **Fitness Evaluation**: Fitness scores are calculated using machine learning models and domain-specific metrics, considering:
   - Probabilities predicted by the model.
   - Custom penalties for specific data regions (e.g., trapped, borderline, or inland samples).
3. **Crossover**: Features from two parent chromosomes are combined to create new samples using uniform crossover, with probability-driven feature selection.
4. **Mutation**: Adds diversity by modifying features based on predefined rates.
5. **Selection**: The best chromosomes are selected for the next generation based on fitness scores.

### Clustering Algorithm
Clustering plays a vital role in grouping data points to:
- Improve synthetic sample relevance by operating within clusters.
- Identify specific regions (e.g., trapped or borderline) for targeted oversampling.
- Facilitate efficient handling of high-dimensional datasets.

**CFSFDP Clustering**: This algorithm uses the Hybrid Entropy-Enhanced Metric (HEEM) to measure distances, combining numerical and categorical feature handling with entropy-based weighting for better cluster formation.

### Hybrid Entropy-Enhanced Metric (HEEM)
HEEM is a custom metric that:
- Computes numerical distances using scaled differences.
- Calculates categorical distances weighted by entropy, emphasizing feature variance.
- Combines these distances to form a hybrid metric for clustering and fitness evaluation.

### Imputation Techniques
To handle missing data, the repository integrates:
- **KNN Imputer**: Uses K-Nearest Neighbors to fill missing values.
- **MICE**: Multiple Imputation by Chained Equations iteratively predicts and fills missing values based on feature correlations.

## Workflow
1. **Preprocessing**: Missing values are imputed, and features are normalized.
2. **Clustering**: Data is grouped into clusters using HEEM-based clustering.
3. **Genetic Algorithm**: Within each cluster, GA generates synthetic samples tailored to minority regions.
4. **Integration**: New samples are combined with the original dataset for training machine learning models.
