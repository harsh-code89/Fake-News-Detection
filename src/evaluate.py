import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add project root to Python path

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from src.preprocess import load_and_preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model():
    """
    Load the saved model and evaluate it using accuracy score and confusion matrix.
    """
    # Load the saved model
    model_path = 'c:/Users/ASUS/Desktop/fake-news-detection-nlp/models/saved_model.pkl'  # Adjusted to absolute path
    model = joblib.load(model_path)

    # Load and preprocess the data
    file_path = 'c:/Users/ASUS/Desktop/fake-news-detection-nlp/data/fake_news.csv'  # Adjusted to absolute path
    _, X_test, _, y_test = load_and_preprocess_data(file_path)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot a confusion matrix using Seaborn heatmap.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    evaluate_model()
