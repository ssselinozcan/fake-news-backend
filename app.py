from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Fake News Detection API")

# Load model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

class NewsRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: NewsRequest):
    text_vectorized = vectorizer.transform([request.text])
    prediction = model.predict(text_vectorized)[0]

    label = "fake" if prediction == 0 else "real"

    return {"prediction": label}
