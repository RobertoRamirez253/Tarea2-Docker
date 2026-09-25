from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Mobile Price Predictor")

# Cargar artefactos generados
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

price_classes = {
    0: "Económico (Low Cost)",
    1: "Gama Media-Baja",
    2: "Gama Media-Alta",
    3: "Gama Alta / Flagship"
}

class MobileFeatures(BaseModel):
    battery_power: float
    blue: int
    clock_speed: float
    dual_sim: int
    fc: float
    four_g: int
    int_memory: float
    m_dep: float
    mobile_wt: float
    n_cores: int
    pc: float
    px_height: float
    px_width: float
    ram: float
    sc_h: float
    sc_w: float
    talk_time: float
    three_g: int
    touch_screen: int
    wifi: int

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(features: MobileFeatures):
    try:
        input_data = np.array([[
            features.battery_power, features.blue, features.clock_speed,
            features.dual_sim, features.fc, features.four_g, features.int_memory,
            features.m_dep, features.mobile_wt, features.n_cores, features.pc,
            features.px_height, features.px_width, features.ram, features.sc_h,
            features.sc_w, features.talk_time, features.three_g,
            features.touch_screen, features.wifi
        ]])
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        return {
            "rango_precio_id": int(prediction),
            "categoria": price_classes[int(prediction)]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
