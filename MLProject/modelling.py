import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# DAGSHUB + MLFLOW INIT
# dagshub.init(
#     repo_owner="anjuan14",
#     repo_name="heart-disease-mlflow",
#     mlflow=True
# )

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Heart Disease Modelling - Advance")


# LOAD DATA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "heart_disease_preprocessed.csv")

df = pd.read_csv(DATA_PATH)

# TARGET & FEATURES (ANTI-LEAKAGE)
y = df["Heart Disease Status"]
X = df.drop(columns=["Heart Disease Status", "Age_Bin"])


# ======================
# TRAIN TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# MODEL & TUNING
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 0],
    "min_samples_split": [2, 5]
}

model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1
)


# MLFLOW RUN
with mlflow.start_run():

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # METRICS
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # LOG PARAMS & METRICS
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # SAVE MODEL
    joblib.dump(best_model, "best_model.pkl")
    mlflow.log_artifact("best_model.pkl")
    
    #MAKING MODEL FOLDER 
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="HeartDiseaseRF"
    )

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    # FEATURE SUMMARY
    X.describe().to_csv("feature_summary.csv")
    mlflow.log_artifact("feature_summary.csv")

    print("ADVANCE RUN SUCCESS")
    print("F1:", f1)



print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(y_test.value_counts())
