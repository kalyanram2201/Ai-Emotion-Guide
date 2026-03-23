from fastapi import FastAPI
import joblib
import pandas as pd
from scipy.sparse import csr_matrix

from src.preprocess import preprocess, combine
from src.decision_engine import decide_action

from src.data_loader import load_data
app = FastAPI()

# ---------- LOAD MODELS ----------
clf = joblib.load("models/emotion_model.pkl")
reg = joblib.load("models/intensity_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")


# ---------- PREDICT FUNCTION ----------
def predict_single(data):
    df = pd.DataFrame([data])

    df = preprocess(df)

    # Text
    x_text = vectorizer.transform(df['clean_text'])

    # Metadata
    numeric_col = ['sleep_hours', 'stress_level', 'energy_level', 'duration_min']
    cat_col = [
        'ambience_type',
        'time_of_day',
        'previous_day_mood',
        'face_emotion_hint',
        'reflection_quality'
    ]

    df[numeric_col] = df[numeric_col].fillna(0)
    df[cat_col] = df[cat_col].fillna("missing")

    x_num = scaler.transform(df[numeric_col])
    x_num = csr_matrix(x_num)

    x_cat = encoder.transform(df[cat_col])

    _, x = combine(x_text, x_text, x_num, x_num, x_cat, x_cat)

    # Predictions
    pred_state = label_encoder.inverse_transform(clf.predict(x))[0]
    pred_intensity = reg.predict(x)[0]
    pred_intensity = round(pred_intensity)

    probs = clf.predict_proba(x)
    confidence = float(probs.max())

    # Decision
    what, when = decide_action(
        pred_state,
        pred_intensity,
        data['stress_level'],
        data['energy_level'],
        data['time_of_day']
    )

    message = f"You seem {pred_state} right now. A good next step would be {what.replace('_', ' ')}."

    return {
        "predicted_state": pred_state,
        "predicted_intensity": pred_intensity,
        "confidence": confidence,
        "what_to_do": what,
        "when_to_do": when,
        "message": message
    }


# ---------- API ROUTE ----------
@app.post("/predict")
def predict(data: dict):
    return predict_single(data)