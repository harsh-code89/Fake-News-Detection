import streamlit as st
import joblib
from src.evaluate import plot_confusion_matrix

# Load the trained model and TF-IDF vectorizer
model_path = 'models/saved_model.pkl'
vectorizer_path = 'models/tfidf_vectorizer.pkl'

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Streamlit app layout
st.title("Fake News Detection Dashboard")
st.write("Enter the news text below to check if it's real or fake.")

# Sidebar for additional options
if st.sidebar.checkbox("Show Confusion Matrix"):
    # Dummy data for demonstration
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1]
    st.write("Confusion Matrix:")
    plot_confusion_matrix(y_true, y_pred)

# User input
user_input = st.text_area("News Text", height=200)

if st.button("Predict"):
    if user_input.strip():
        # Transform the input text using the TF-IDF vectorizer
        input_vectorized = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(input_vectorized)[0]

        # Display result
        if prediction == 1:
            st.success("The news is Real.")
        else:
            st.error("The news is Fake.")
    else:
        st.warning("Please enter some text to make a prediction.")
