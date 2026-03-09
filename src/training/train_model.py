import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create experiment
mlflow.set_experiment("stock_prediction_experiment")

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():

    n_estimators = 100

    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # log parameter
    mlflow.log_param("n_estimators", n_estimators)

    # log metric
    mlflow.log_metric("accuracy", accuracy)

   # log model artifact and register model
    mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="stock_prediction_model"
)

    print("Model trained with accuracy:", accuracy)