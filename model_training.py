"""
Model training script for the UCI Bank Marketing dataset.

This script:
- Loads the bank-additional-full.csv file from UCI
- Preprocesses numeric and categorical columns
- Trains 6 different classification models
- Evaluates them using the required metrics
- Saves the trained models and evaluation results to a model/ folder
"""

import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the Bank Marketing dataset.

    Expects the UCI `bank-additional-full.csv` file with `;` as separator.
    """
    df = pd.read_csv(csv_path, sep=";")
    return df


def prepare_features(df: pd.DataFrame):
    """
    Split dataframe into features X and target y.

    Target column is `y` (yes/no). It is mapped to 1/0.
    """
    X = df.drop("y", axis=1)
    y = (df["y"] == "yes").astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create a ColumnTransformer for numeric and categorical features.
    """
    # Standard numeric and categorical columns for bank-additional-full.csv
    numeric_features = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]

    categorical_features = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
    ]

    # Safety check: keep only columns that actually exist
    numeric_features = [c for c in numeric_features if c in X.columns]
    categorical_features = [c for c in categorical_features if c in X.columns]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def build_models(preprocessor: ColumnTransformer):
    """
    Create a dict of model name -> sklearn Pipeline.
    """
    base_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    pipelines = {}
    for name, model in base_models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        pipelines[name] = pipe

    return pipelines


def evaluate_model(y_true, y_pred, y_score=None):
    """
    Return a dict with all required metrics for a single model.
    """
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)

    if y_score is not None:
        try:
            metrics["AUC"] = roc_auc_score(y_true, y_score)
        except ValueError:
            # AUC can fail if only one class is present in y_true
            metrics["AUC"] = np.nan
    else:
        metrics["AUC"] = np.nan

    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

    return metrics


def main():
    # 1. Load data (make sure this file is present in the same folder)
    csv_path = "bank-additional-full.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find '{csv_path}'. "
            "Please place the UCI bank-additional-full.csv file "
            "in the same folder as this script."
        )

    print("Loading data...")
    df = load_data(csv_path)
    print(f"Dataset shape: {df.shape}")

    # 2. Prepare X and y
    X, y = prepare_features(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} rows")
    print(f"Test size: {X_test.shape[0]} rows")

    # 4. Preprocessor
    preprocessor = build_preprocessor(X_train)

    # 5. Build models
    pipelines = build_models(preprocessor)

    # Ensure model directory exists
    os.makedirs("model", exist_ok=True)

    # 6. Train, evaluate, save
    results = []

    for name, pipe in pipelines.items():
        print("\n" + "=" * 60)
        print(f"Training model: {name}")
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_score = pipe.predict_proba(X_test)[:, 1]
        else:
            y_score = None

        metrics = evaluate_model(y_test, y_pred, y_score)
        metrics_row = {"ML Model Name": name}
        metrics_row.update(metrics)
        results.append(metrics_row)

        # Save pipeline
        short_name = name.lower().replace(" ", "_")
        model_path = os.path.join("model", f"{short_name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)
        print(f"Saved model to {model_path}")
        print(
            "Metrics:",
            {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        )

    # 7. Save metrics table
    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    results_df = results_df.round(4)
    results_csv_path = os.path.join("model", "model_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print("\nSaved metrics to", results_csv_path)
    print(results_df)

    # 8. Save test data for Streamlit app
    test_df = X_test.copy()
    test_df["y"] = y_test.values
    test_csv_path = os.path.join("model", "test_data.csv")
    test_df.to_csv(test_csv_path, index=False)
    print("Saved test data to", test_csv_path)

    print("\nAll models trained and saved successfully.")


if __name__ == "__main__":
    main()
