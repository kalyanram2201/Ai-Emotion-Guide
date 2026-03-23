import pandas as pd
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix

# ---------- STOPWORDS ----------
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))


# ---------- TEXT CLEANING ----------
def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)


# ---------- PREPROCESS ----------
def preprocess(df):
    df = df.copy()
    df['clean_text'] = df['journal_text'].apply(clean)
    return df


# ---------- TEXT VECTOR ----------
def vectorize_text(train_df, test_df):
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )

    x_train_text = vectorizer.fit_transform(train_df['clean_text'])
    x_test_text = vectorizer.transform(test_df['clean_text'])

    return x_train_text, x_test_text, vectorizer


# ---------- METADATA ----------
def process_meta(train, test):
    scaler = StandardScaler()

    numeric_col = ['sleep_hours', 'stress_level', 'energy_level', 'duration_min']

    train[numeric_col] = train[numeric_col].fillna(train[numeric_col].mean())
    test[numeric_col] = test[numeric_col].fillna(test[numeric_col].mean())

    x_train_num = scaler.fit_transform(train[numeric_col])
    x_test_num = scaler.transform(test[numeric_col])

    x_train_num = csr_matrix(x_train_num)
    x_test_num = csr_matrix(x_test_num)

    cat_col = [
        'ambience_type',
        'time_of_day',
        'previous_day_mood',
        'face_emotion_hint',
        'reflection_quality'
    ]

    train[cat_col] = train[cat_col].fillna("missing")
    test[cat_col] = test[cat_col].fillna("missing")

    encoder = OneHotEncoder(handle_unknown='ignore')

    x_train_cat = encoder.fit_transform(train[cat_col])
    x_test_cat = encoder.transform(test[cat_col])

    return x_train_num, x_test_num, x_train_cat, x_test_cat, scaler, encoder


# ---------- LABELS ----------
def encode_labels(train_df):
    le = LabelEncoder()
    y_state = le.fit_transform(train_df['emotional_state'])
    y_intensity = train_df['intensity']
    return y_state, y_intensity, le


# ---------- COMBINE ----------
def combine(x_train_text, x_test_text, x_train_num, x_test_num, x_train_cat, x_test_cat):
    x_train = hstack([x_train_text, x_train_num, x_train_cat])
    x_test = hstack([x_test_text, x_test_num, x_test_cat])
    return x_train, x_test