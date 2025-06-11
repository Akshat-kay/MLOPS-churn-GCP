from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Load model at startup
try:
    model = joblib.load("model.joblib")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    model = None

class ChurnRequest(BaseModel):
    data: dict

# Add health check endpoint (MANDATORY for Cloud Run)
@app.get("/healthz")
async def health_check():
    return {"status": "ready" if model else "loading failed"}

@app.post("/predict")
def predict(req: ChurnRequest):
    if not model:
        return {"error": "Model not loaded"}, 503
    
    try:
        df = pd.DataFrame([req.data])
        df = pd.get_dummies(df)
        prediction = model.predict(df)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        return {"error": str(e)}, 400
