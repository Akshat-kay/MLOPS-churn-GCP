from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

class ChurnRequest(BaseModel):
    data: dict

app = FastAPI()
model = joblib.load("model.joblib")

@app.post("/predict")
def predict(req: ChurnRequest):
    df = pd.DataFrame([req.data])
    df = pd.get_dummies(df)
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}

