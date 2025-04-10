from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.exceptions import HTTPException
import pandas as pd
import joblib
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_dir = os.path.dirname(base_dir)
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static")
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../templates")
models_dir = os.path.join(project_dir, "models") 

app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=templates_dir)

model = joblib.load(os.path.join(models_dir, "gradient_boosting_model.pkl"))
scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

class StrokeRiskInput(BaseModel):
    chest_pain: bool = False
    shortness_of_breath: bool = False
    irregular_heartbeat: bool = False
    fatigue_weakness: bool = False
    dizziness: bool = False
    swelling_edema: bool = False
    pain_in_neck_jaw_shoulder_back: bool = False
    excessive_sweating: bool = False
    persistent_cough: bool = False
    nausea_vomiting: bool = False
    high_blood_pressure: bool = False
    chest_discomfort_activity: bool = False
    cold_hands_feet: bool = False
    snoring_sleep_apnea: bool = False
    anxiety_feeling_of_doom: bool = False
    age: int

    def prepare_input(self):
        input_dict = {
            'Chest Pain': int(self.chest_pain),
            'Shortness of Breath': int(self.shortness_of_breath),
            'Irregular Heartbeat': int(self.irregular_heartbeat),
            'Fatigue & Weakness': int(self.fatigue_weakness),
            'Dizziness': int(self.dizziness),
            'Swelling (Edema)': int(self.swelling_edema),
            'Pain in Neck/Jaw/Shoulder/Back': int(self.pain_in_neck_jaw_shoulder_back),
            'Excessive Sweating': int(self.excessive_sweating),
            'Persistent Cough': int(self.persistent_cough),
            'Nausea/Vomiting': int(self.nausea_vomiting),
            'High Blood Pressure': int(self.high_blood_pressure),
            'Chest Discomfort (Activity)': int(self.chest_discomfort_activity),
            'Cold Hands/Feet': int(self.cold_hands_feet),
            'Snoring/Sleep Apnea': int(self.snoring_sleep_apnea),
            'Anxiety/Feeling of Doom': int(self.anxiety_feeling_of_doom),
            'Age': self.age
        }

        raw_df = pd.DataFrame([input_dict])

        for col in scaler.feature_names_in_:
            if col not in raw_df.columns:
                raw_df[col] = 0

        raw_df = raw_df[scaler.feature_names_in_]

        scaled_input = scaler.transform(raw_df)
        return pd.DataFrame(scaled_input, columns=scaler.feature_names_in_)

@app.post("/predict")
async def predict_stroke_risk(input_data: StrokeRiskInput):
    try:
        input_df = input_data.prepare_input()
        
        prediction = model.predict(input_df)[0]
        risk_category = "High Risk" if prediction >= 50 else "Low Risk"
        
        return {
            "prediction": round(float(prediction), 2),
            "risk_category": risk_category,
            "age": input_data.age
        }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/result/", response_class=HTMLResponse)
async def result_page(request: Request, prediction: float, risk_category: str, age: int):
    return templates.TemplateResponse(
        "result.html", 
        {
            "request": request, 
            "prediction": prediction,
            "risk_category": risk_category,
            "age": age
        }
    )