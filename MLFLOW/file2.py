import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import dagshub

dagshub.init(repo_owner="Ghanshyam", repo_name="mlflow-Dagshub", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Ghanshyam1904/mlflow-Dagshub.mlflow")  # check repo_owner spelling!

wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
max_depth, n_estimators = 10, 10

mlflow.set_experiment("Mlflow-Dagshub")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_params({"max_depth": max_depth, "n_estimators": n_estimators})

    cn = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cn, annot=True, fmt="d", cmap="Blues",
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")
    try:
        mlflow.log_artifact(__file__)
    except NameError:
        pass  # Ignore if __file__ is not available (e.g., in notebook)

    mlflow.set_tags({"Author": "Ghanshyam", "Project": "wine classifier"})
    mlflow.sklearn.log_model(rf, "RandomForestClassifier")
    print(accuracy)
