import joblib
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess_data


def train_and_save_model():
    """
    Train a PassiveAggressiveClassifier using TF-IDF features and save the model.
    """
    # Load and preprocess the data
    file_path = '../data/fake_news.csv'  # Adjust the path as needed
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    # Initialize the PassiveAggressiveClassifier
    model = PassiveAggressiveClassifier(max_iter=50, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model
    model_path = '../models/saved_model.pkl'  # Adjust the path as needed
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_and_save_model()
