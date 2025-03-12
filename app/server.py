#pip install fastapi
#pip install uvicorn

# Librairies
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris

iris = load_iris()

# Model Loading
loaded_model = load('./app/logreg.joblib')

# An instance of the fastAPI
app = FastAPI()

# Define request class
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

# End point preparation
@app.post("/predict") # local : http://127.0.0.1:8000/predict

# Prediction function
def predict(data : request_body):
    # Formating the data
    new_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Prediction
    class_idx = loaded_model.predict(new_data)[0]

    # iris name_class is returned
    return {'class' : iris.target_names[class_idx]}
