import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model():
    # Load processed data
    train = pd.read_csv("diabetes_preprocessing/diabetes_train.csv")
    test = pd.read_csv("diabetes_preprocessing/diabetes_test.csv")

    X_train = train.drop("Outcome", axis=1)
    y_train = train["Outcome"]
    X_test = test.drop("Outcome", axis=1)
    y_test = test["Outcome"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(1)
    )

    print(f"\nResults:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")

if __name__ == "__main__":
    train_model()