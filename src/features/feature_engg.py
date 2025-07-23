import pandas as pd 
import os
import yaml
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure logs directory exists BEFORE configuring logging
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='logs/feature_engg.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def main():
    try:
        # Load parameters from params.yaml
        try:
            with open('params.yaml', 'r') as file:
                params = yaml.safe_load(file)
            max_features = params['feature_engg']['max_features']
            logging.info("Loaded max_features parameter: %s", max_features)
        except Exception as e:
            logging.error("Failed to load params.yaml: %s", e)
            raise

        # Load processed train and test data
        try:
            train_data = pd.read_csv("data/processed/train.csv").dropna(subset=['content'])
            test_data = pd.read_csv("data/processed/test.csv").dropna(subset=['content'])
            logging.info("Loaded train and test data with shapes: %s, %s", train_data.shape, test_data.shape)
        except Exception as e:
            logging.error("Failed to load processed data: %s", e)
            raise

        # Extract features and labels from train and test data
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        # Apply Bag of Words (CountVectorizer)
        try:
            vectorizer = TfidfVectorizer(max_features=max_features)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf= vectorizer.transform(X_test)
            logging.info("Bag of Words transformation completed.")
        except Exception as e:
            logging.error("Error during Bag of Words transformation: %s", e)
            raise

        # Convert the feature vectors to DataFrames for easier handling
        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test   

        # Save the processed feature data to CSV files
        try:
            os.makedirs("data/interim", exist_ok=True)  # Ensure the directory exists
            train_df.to_csv("data/interim/train_tfidf.csv", index=False)
            test_df.to_csv("data/interim/test_tfidf.csv", index=False)
            logging.info("Feature data saved to data/interim/")
        except Exception as e:
            logging.error("Failed to save feature data: %s", e)
            raise

    except Exception as e:
        logging.critical("Unhandled exception in feature engineering: %s", e)
        raise

if __name__ == "__main__":
    main()