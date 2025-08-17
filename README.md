# Fake News Detection NLP

## Overview
The Fake News Detection NLP project aims to classify news articles as real or fake using natural language processing techniques. This project leverages machine learning models to analyze text data and make predictions based on patterns in the content.

## Folder Structure
```
fake-news-detection-nlp/
├── data/                # Contains the dataset (e.g., fake_news.csv)
├── notebooks/           # Jupyter notebooks for exploratory data analysis (EDA)
├── src/                 # Source code for preprocessing, training, and evaluation
│   ├── preprocess.py    # Data loading and preprocessing
│   ├── train_model.py   # Model training script
│   └── evaluate.py      # Model evaluation script
├── models/              # Directory to store trained models (e.g., saved_model.pkl)
├── utils/               # Utility functions (e.g., helpers.py)
├── requirements.txt     # List of required Python packages
├── README.md            # Project documentation
└── .gitignore           # Files and directories to ignore in version control
```

## Techniques Used
- **TF-IDF Vectorization**: Converts text data into numerical features by analyzing term frequency and inverse document frequency.
- **PassiveAggressiveClassifier**: A machine learning algorithm used for binary classification tasks, particularly effective for text classification.

## Dataset
The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/). It contains labeled news articles with columns for text and labels (real or fake).

## Instructions to Run the Project

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd fake-news-detection-nlp
   ```

2. **Install Dependencies**
   Ensure you have Python installed. Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**
   Place the `fake_news.csv` file in the `data/` directory.

4. **Run Preprocessing and Training**
   Execute the training script to preprocess the data and train the model:
   ```bash
   python src/train_model.py
   ```

5. **Evaluate the Model**
   Run the evaluation script to test the model and view metrics:
   ```bash
   python src/evaluate.py
   ```

## Contributions
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License.
