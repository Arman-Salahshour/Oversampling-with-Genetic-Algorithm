from constants import *
from matplotlib import pyplot as plt



def plot_roc_auc(df):
    """
    Plots the ROC AUC curves for models based on input DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing TPR, FPR, and AUC values for models.
    
    Returns:
    None
    """
    # Initialize a figure for the ROC plot
    fig1 = plt.figure(figsize=[12, 12])
    ax1 = fig1.add_subplot(111, aspect='equal')

    # Plot the diagonal (random chance baseline)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')

    # Iterate through each column in the DataFrame to plot individual ROC curves
    for col in df.columns:
        # Retrieve TPR, FPR, and AUC values for the model
        mean_tpr = df.loc['tpr', [col]].to_numpy()[0]
        mean_fpr = df.loc['fpr', [col]].to_numpy()[0]
        mean_auc = df.loc['AUC', [col]]

        # Plot the ROC curve with the corresponding AUC label
        plt.plot(mean_fpr, mean_tpr, label=f'{str(col)}: AUC = {round(float(mean_auc), 2)}', lw=2, alpha=1)

    # Add labels, legend, and title for the plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")

    # Annotate regions for accuracy
    plt.text(0.32, 0.7, 'More accurate area', fontsize=12)
    plt.text(0.63, 0.4, 'Less accurate area', fontsize=12)

    # Show the plot
    plt.show()



def train_model(train_set, test_set, model, random_state=2021, deep=False, stacking=False):
    """
    Trains a given model on training data and evaluates it on test data.

    Parameters:
    train_set (tuple): Tuple containing training features (X_train) and labels (Y_train).
    test_set (tuple): Tuple containing test features (X_test) and labels (Y_test).
    model: Machine learning or deep learning model to be trained.
    random_state (int): Seed for reproducibility. Default is 2021.
    deep (bool): Flag to indicate if a deep learning model is used. Default is False.
    stacking (bool): Flag to indicate if a stacking model is used. Default is False.

    Returns:
    tuple: Predictions and probability predictions for the test set.
    """
    # Extract train and test sets
    X_train, Y_train = train_set
    X_test, Y_test = test_set

    if deep:
        # Configure learning rate scheduler and compile the model
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-1 / 10**(epoch / 50))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                      loss='binary_crossentropy', 
                      metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        # Train the deep learning model
        model.fit(X_train, Y_train, epochs=40, validation_data=test_set, verbose=False)

    if stacking:
        # Train stacking models
        model.fit(X_train, Y_train)

    if deep or stacking:
        # Generate predictions for deep or stacking models
        probability_predictions = model.predict(X_test)
        predictions = copy.deepcopy(probability_predictions)
        predictions[predictions >= 0.5] = 1.  # Apply threshold
        predictions[predictions < 0.5] = 0.
        predictions = predictions.flatten()
    else:
        # Train traditional ML models and generate predictions
        model.fit(X_train, Y_train)
        probability_predictions = model.predict_proba(X_test)[:, 1]
        predictions = probability_predictions >= 0.46  # Custom threshold
        predictions = predictions.astype(float).flatten()

    return predictions, probability_predictions
