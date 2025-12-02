import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model, scaler, target_names
    model_path = Model.get_model_path('iris_model')
    model, scaler = joblib.load(model_path)
    target_names = ['setosa', 'versicolor', 'virginica']

def run(data):
    try:
        data = json.loads(data)['data']
        arr = np.array(data)
        arr = scaler.transform(arr)
        preds = model.predict(arr)
        classes = [target_names[p] for p in preds]
        return {"prediction": classes}
    except Exception as e:
        return {"error": str(e)}
