import re
import logging
import matplotlib.pyplot as plt
import seaborn as sns


def clean_text(text):
    """
    Clean the input text by removing non-alphanumeric characters and converting to lowercase.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()


def setup_logging(log_file='app.log'):
    """
    Set up logging configuration.

    Args:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def plot_confusion_matrix(cm, labels):
    """
    Plot a confusion matrix using Seaborn heatmap.

    Args:
        cm (array-like): Confusion matrix.
        labels (list): List of class labels.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
