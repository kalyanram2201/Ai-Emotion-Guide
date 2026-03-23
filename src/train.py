import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_absolute_error

from preprocess import (
    preprocess,
    vectorize_text,
    process_meta,
    combine,
    encode_labels
)
from data_loader import load_data


# ---------- LOAD ----------
train, test = load_data()

train = preprocess(train)
test = preprocess(test)

# ---------- FEATURES ----------
x_train_text, x_test_text, vectorizer = vectorize_text(train, test)

x_train_num, x_test_num, x_train_cat, x_test_cat, scaler, encoder = process_meta(train, test)

x, _ = combine(
    x_train_text, x_test_text,
    x_train_num, x_test_num,
    x_train_cat, x_test_cat
)

y_state, y_intensity, label_encoder = encode_labels(train)


# ---------- SPLIT ----------
x_train, x_val, y_state_train, y_state_val = train_test_split(
    x, y_state, test_size=0.2, random_state=42
)

_, _, y_int_train, y_int_val = train_test_split(
    x, y_intensity, test_size=0.2, random_state=42
)


# ---------- EMOTION MODEL ----------
clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

clf.fit(x_train, y_state_train)

y_pred_state = clf.predict(x_val)

print("\n===== Emotion Model =====")
print("Accuracy:", accuracy_score(y_state_val, y_pred_state))
print(classification_report(y_state_val, y_pred_state))


# ---------- INTENSITY MODEL (REGRESSION ✅) ----------
reg = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

reg.fit(x_train, y_int_train)

y_pred_int = reg.predict(x_val)

print("\n===== Intensity Model =====")
print("MAE:", mean_absolute_error(y_int_val, y_pred_int))


# ---------- SAVE ----------
joblib.dump(clf, "models/emotion_model.pkl")
joblib.dump(reg, "models/intensity_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoder, "models/encoder.pkl")

print("\nModels saved successfully!")