import joblib
import pandas as pd
from scipy.sparse import csr_matrix

from preprocess import preprocess, combine
from data_loader import load_data
from decision_engine import decide_action


# ---------- LOAD ----------
clf = joblib.load("models/emotion_model.pkl")
reg = joblib.load("models/intensity_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")


# ---------- DATA ----------
train, test = load_data()

train = preprocess(train)
test = preprocess(test)


# ---------- TEXT ----------
x_train_text = vectorizer.transform(train['clean_text'])
x_test_text = vectorizer.transform(test['clean_text'])


# ---------- METADATA ----------
numeric_col = ['sleep_hours', 'stress_level', 'energy_level', 'duration_min']
cat_col = [
    'ambience_type',
    'time_of_day',
    'previous_day_mood',
    'face_emotion_hint',
    'reflection_quality'
]

train[numeric_col] = train[numeric_col].fillna(train[numeric_col].mean())
test[numeric_col] = test[numeric_col].fillna(test[numeric_col].mean())

x_train_num = scaler.transform(train[numeric_col])
x_test_num = scaler.transform(test[numeric_col])

x_train_num = csr_matrix(x_train_num)
x_test_num = csr_matrix(x_test_num)

train[cat_col] = train[cat_col].fillna("missing")
test[cat_col] = test[cat_col].fillna("missing")

x_train_cat = encoder.transform(train[cat_col])
x_test_cat = encoder.transform(test[cat_col])


# ---------- COMBINE ----------
_, x_test = combine(
    x_train_text, x_test_text,
    x_train_num, x_test_num,
    x_train_cat, x_test_cat
)


# ---------- PREDICT ----------
pred_state_encoded = clf.predict(x_test)
pred_state = label_encoder.inverse_transform(pred_state_encoded)

pred_intensity = reg.predict(x_test)
pred_intensity = pred_intensity.round().clip(1, 5)


# ---------- CONFIDENCE ----------
probs = clf.predict_proba(x_test)
confidence = probs.max(axis=1)

uncertain_flag = (confidence < 0.6).astype(int)


# ---------- SMART RULES ----------
for i in range(len(pred_intensity)):

    if test.iloc[i]['stress_level'] >= 4 and pred_state[i] in ["overwhelmed", "mixed"]:
        pred_intensity[i] = min(5, pred_intensity[i] + 1)

    elif test.iloc[i]['energy_level'] <= 2:
        pred_intensity[i] = max(1, pred_intensity[i] - 1)

    if confidence[i] < 0.5:
        pred_intensity[i] = 3


# ---------- DECISION ----------
actions = []
timings = []

for i in range(len(test)):
    what, when = decide_action(
        pred_state[i],
        pred_intensity[i],
        test.iloc[i]['stress_level'],
        test.iloc[i]['energy_level'],
        test.iloc[i]['time_of_day']
    )

    actions.append(what)
    timings.append(when)


# ---------- OUTPUT ----------
output = pd.DataFrame({
    "id": test["id"],
    "predicted_state": pred_state,
    "predicted_intensity": pred_intensity,
    "confidence": confidence,
    "uncertain_flag": uncertain_flag,
    "what_to_do": actions,
    "when_to_do": timings
})

output.to_csv("outputs/predictions.csv", index=False)

print("\nPredictions saved!")
print(output.head())