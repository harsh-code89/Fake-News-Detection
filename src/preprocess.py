import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re


def load_and_preprocess_data(file_path):
    """
    Load the dataset, clean the text data, apply TF-IDF vectorization, and return train-test split.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Check if required columns exist
    if 'text' not in data.columns or 'label' not in data.columns:
        raise ValueError("The dataset must contain 'text' and 'label' columns.")

    # Clean the text data
    def clean_text(text):
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        return text

    data['cleaned_text'] = data['text'].apply(clean_text)

    # Apply TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['cleaned_text']).toarray()
    y = data['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
