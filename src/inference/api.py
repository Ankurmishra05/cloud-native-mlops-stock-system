import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
import pandas as pd

# Connect to MLflow server
mlflow.set_tracking_uri("http://host.docker.internal:5000")

app = FastAPI()

MODEL_NAME = "stock_prediction_model"

# Load latest model version from registry
model = mlflow.pyfunc.load_model("models/model")

@app.get("/")
def home():
    return {"message": "ML inference API running"}

@app.post("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):

    data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ],
    )

    prediction = model.predict(data)

    return {"prediction": int(prediction[0])}