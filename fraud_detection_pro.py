import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


RANDOM_STATE = 42


# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------
def load_data(path):
    print("Loading dataset...")
    data = pd.read_csv(path)
    print(f"Dataset shape: {data.shape}")
    return data


# --------------------------------------------------
# 2. Split Data
# --------------------------------------------------
def split_data(data):
    X = data.drop("Class", axis=1)
    y = data["Class"]

    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )


# --------------------------------------------------
# 3. Build Pipeline
# --------------------------------------------------
def build_pipeline():
    pipeline = Pipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    return pipeline


# --------------------------------------------------
# 4. Hyperparameter Tuning
# --------------------------------------------------
def tune_model(pipeline, X_train, y_train):
    param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=5,
        scoring="f1",
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    search.fit(X_train, y_train)
    print("Best Parameters:", search.best_params_)
    return search.best_estimator_


# --------------------------------------------------
# 5. Evaluation
# --------------------------------------------------
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()


# --------------------------------------------------
# 6. Save Model
# --------------------------------------------------
def save_model(model, filename="fraud_model.pkl"):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")


# --------------------------------------------------
# Main Execution
# --------------------------------------------------
def main():
    data = load_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test = split_data(data)

    pipeline = build_pipeline()
    best_model = tune_model(pipeline, X_train, y_train)

    evaluate_model(best_model, X_test, y_test)
    save_model(best_model)


if __name__ == "__main__":
    main()
