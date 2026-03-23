# 🌿 AI Emotional Guidance System

## 🚀 Overview

This project builds an intelligent system that goes beyond prediction — it **understands human emotional state and guides users toward better mental actions**.

The system processes:

* 📝 Noisy, short journal reflections
* 🧠 Contextual signals (sleep, stress, energy, time)

And produces:

* Emotional state
* Intensity (1–5)
* Action recommendation (what to do)
* Timing (when to do it)
* Confidence & uncertainty

---

## 🧠 System Pipeline

```
User Input → Preprocessing → ML Models → Decision Engine → Output
```

---

## ⚙️ Features

### 1️⃣ Emotional Understanding

* Model: **XGBoost Classifier**
* Predicts emotional state:

  * calm, restless, neutral, focused, mixed, overwhelmed

---

### 2️⃣ Intensity Prediction

* Model: **XGBoost Regressor**
* Treated as regression (continuous emotion)
* Output mapped to range **[1–5]**

---

### 3️⃣ Decision Engine (Core)

Rule-based system that decides:

#### ✔️ What to do:

* rest, movement, breathing, journaling, deep work, etc.

#### ✔️ When to do:

* now, within_15_min, later_today, tonight, tomorrow_morning

Uses:

* predicted emotion
* intensity
* stress level
* energy level
* time of day

---

### 4️⃣ Uncertainty Modeling

* Confidence = max probability from classifier
* `uncertain_flag = 1` if confidence < 0.6

👉 System knows when it is unsure.

---

### 5️⃣ API (FastAPI)

* Real-time prediction endpoint:

```
POST /predict
```

---

### 6️⃣ UI (Streamlit)

* Interactive interface for user input
* Displays:

  * emotion
  * intensity
  * recommendation
  * confidence
  * AI guidance message

---

## 📊 Results

| Task                 | Metric        |
| -------------------- | ------------- |
| Emotion Prediction   | ~66% Accuracy |
| Intensity Prediction | ~1.27 MAE     |

---

## 🔬 Feature Engineering

### 📝 Text Features

* TF-IDF vectorization
* Unigrams + Bigrams
* Stopword removal

---

### 📊 Metadata Features

* Numerical → StandardScaler
* Categorical → OneHotEncoder

---

## ⚖️ Ablation Study

| Model Type      | Performance |
| --------------- | ----------- |
| Text Only       | Lower       |
| Text + Metadata | Higher      |

👉 Metadata improves robustness under noisy inputs.

---

## ❌ Error Analysis

Common failure cases:

* Very short text ("ok", "fine")
* Conflicting emotional signals
* Noisy or ambiguous labels

Improvements:

* Rule-based correction
* Metadata weighting
* Uncertainty handling

---

## 📱 Edge / Deployment Plan

* Lightweight models (TF-IDF + XGBoost)
* Model size: < 20MB
* Latency: < 100ms
* Can run on-device (mobile/edge)

---

## 🛠️ Tech Stack

* Python
* scikit-learn
* XGBoost
* FastAPI
* Streamlit
* NLTK

---

## ⚙️ Installation & Setup

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Project

### 1️⃣ Start API

```bash
python -m uvicorn src.api:app --reload
```

### 2️⃣ Start UI

```bash
streamlit run app.py
```

---

## 📌 Example Output

```json
{
  "predicted_state": "calm",
  "predicted_intensity": 3,
  "confidence": 0.37,
  "what_to_do": "rest",
  "when_to_do": "now"
}
```

---

## 💡 Key Insight

> Text provides emotional context, while metadata provides stability.

---

## 🏁 Conclusion

This project demonstrates:

* Handling real-world noisy data
* Reasoning under uncertainty
* Hybrid intelligence (ML + rules)
* Product-oriented thinking (API + UI)

---

## 🌱 Philosophy

> AI should not just understand humans — it should guide them toward a better state.

---

Dream > Innovate > Create
— Team ArvyaX
