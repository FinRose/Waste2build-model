from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("waste2build_model.pkl")

@app.get("/")
def home():
    return {"message": "Waste2Build AI API running"}

@app.post("/predict")
def predict(freq: int, accuracy: float, total_kg: float, purity: float):

    input_df = pd.DataFrame([{
        'frequency_per_month': freq,
        'avg_accuracy_score': accuracy,
        'total_kg_last_month': total_kg,
        'material_purity_rate': purity
    }])

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": "High Potential" if prediction[0] == 1 else "Standard",
        "probability": float(probability)
    }