from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.exceptions import HTTPException
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("../../models/gradient_boosting_model.pkl")  
scaler = joblib.load("../../models/scaler.pkl") 

EXPECTED_FEATURES = [
    "Anxiety/Feeling of Doom", "Chest Discomfort (Activity)", "Chest Pain", "Cold Hands/Feet",
    "Excessive Sweating", "Fatigue/Weakness", "Heart Palpitations", "High Blood Pressure",
    "High Cholesterol", "History of Diabetes", "History of Smoking", "Irregular Heartbeat",
    "Obesity", "Physical Inactivity", "Shortness of Breath", "Swelling in Legs/Ankles",
    "Unhealthy Diet", "Age_Scaled"
]

class StrokeRiskInput(BaseModel):
    Chest_Pain: int
    Shortness_of_Breath: int
    Irregular_Heartbeat: int
    Fatigue_Weakness: int
    Dizziness: int
    Swelling_Edema: int
    High_Blood_Pressure: int
    High_Cholesterol: int
    Diabetes: int
    Family_History_of_Heart_Disease: int
    Smoking: int
    Obesity: int
    Physical_Inactivity: int
    Unhealthy_Diet: int
    High_Stress_Levels: int
    Age: int
    Age_Scaled: float

    def to_model_input(self):
        
        feature_mapping = {
           "Chest Pain": self.Chest_Pain,
           "Shortness of Breath": self.Shortness_of_Breath,
           "Irregular Heartbeat": self.Irregular_Heartbeat,
           "Fatigue & Weakness": self.Fatigue_Weakness, 
           "Dizziness": self.Dizziness, 
           "Swelling in Legs/Ankles": self.Swelling_Edema,
           "High Blood Pressure": self.High_Blood_Pressure,
           "High Cholesterol": self.High_Cholesterol,
           "Diabetes": self.Diabetes, 
           "Family History of Heart Disease": self.Family_History_of_Heart_Disease, 
           "Smoking": self.Smoking,  
           "Obesity": self.Obesity,
           "Physical Inactivity": self.Physical_Inactivity,
           "Unhealthy Diet": self.Unhealthy_Diet,
           "High Stress Levels": self.High_Stress_Levels,  
           "Age": self.Age,  
           "Age_Scaled": self.Age_Scaled
        }
        
        return pd.DataFrame([{name: feature_mapping.get(name, 0) for name in EXPECTED_FEATURES}])


@app.get("/")
def read_root():
    return {"message": "Hello World!"}

@app.post("/predict-risk/")
def predict_stroke_risk(input_data: StrokeRiskInput):
    try:
        input_df = input_data.to_model_input()
        
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)
        
        return {"stroke_risk_percentage": float(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")