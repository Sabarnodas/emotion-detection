import pandas as pd
import pickle
import json
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logging.basicConfig(
    filename='logs/model_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def main():
    try:
        os.makedirs("logs", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        # Load model
        try:
            with open("models/random_forest_model.pkl", "rb") as f:
                model = pickle.load(f)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error("Failed to load model: %s", e)
            raise

        # Load test data
        try:
            test_data = pd.read_csv("data/interim/test_tfidf.csv")
            X_test = test_data.drop(columns=['label']).values
            y_test = test_data['label'].values
            logging.info("Test data loaded successfully with shape: %s", test_data.shape)
        except Exception as e:
            logging.error("Failed to load test data: %s", e)
            raise

        # Predict and calculate metrics
        try:
            y_pred = model.predict(X_test)
            metrics_dict = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred)
            }
            logging.info("Model evaluation metrics calculated.")
        except Exception as e:
            logging.error("Failed to load test data: %s", e)
            raise

        # Predict and calculate metrics
        try:
            y_pred = model.predict(X_test)
            metrics_dict = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred)
            }
            logging.info("Model evaluation metrics calculated.")
        except Exception as e:
            logging.error("Error during prediction or metric calculation: %s", e)
            raise

        # Save metrics
        try:
            with open("reports/metrics.json", "w") as f:
                json.dump(metrics_dict, f, indent=4)
            logging.info("Metrics saved to reports/metrics.json")
        except Exception as e:
            logging.error("Failed to save metrics: %s", e)
            raise

    except Exception as e:
        logging.critical("Unhandled exception in model evaluation: %s", e)
        raise

if __name__ == "__main__":
    main()