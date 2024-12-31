# GeneticOversampler
A Python implementation for synthetic data generation and oversampling using genetic algorithms (GA) to address class imbalance issues in datasets.

## Overview
The `GeneticOversampler` repository leverages genetic algorithms to generate synthetic samples for the minority class in imbalanced datasets. The core approach integrates domain knowledge, hybrid distance metrics, and evolutionary principles to generate high-quality synthetic data points. The algorithm includes customizable fitness functions, crossover techniques, and mutation strategies.

## Features
- **Hybrid Distance Metric (HEEM):** Combines categorical and numerical distance metrics with entropy weighting for accurate similarity computation.
- **Genetic Algorithm for Oversampling:** Generates synthetic samples for minority classes using selection, crossover, and mutation.
- **Advanced Fitness Function:** Incorporates both machine learning model probabilities and domain-specific penalties for trapped, inland, and borderline samples.
- **Imputation Techniques:** Handles missing data using KNN and MICE (Multiple Imputation by Chained Equations).
- **Customizable:** Supports various hyperparameters for GA iterations, mutation rates, and fitness evaluation.
- **Visualization and Evaluation:** Includes tools for plotting ROC-AUC curves and evaluating model performance.

## Algorithm Highlights
### Hybrid Entropy-Enhanced Metric (HEEM)
A hybrid distance metric used to measure similarities between data points:
- Numerical distances are scaled using standard deviations.
- Categorical distances are weighted by entropy values, ensuring proper significance to feature variance.

### Genetic Algorithm for Oversampling
1. **Initialization:** Minority samples are selected as base chromosomes.
2. **Fitness Calculation:** Combines logistic regression probabilities and custom penalties for domain-specific considerations.
3. **Crossover and Mutation:** Creates new samples by combining features from parents with a mutation mechanism to enhance diversity.
4. **Selection:** Retains the best chromosomes based on fitness scores for the next generation.

