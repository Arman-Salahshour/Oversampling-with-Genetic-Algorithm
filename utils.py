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
