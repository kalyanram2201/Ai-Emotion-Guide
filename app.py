import streamlit as st
import requests

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="ArvyaX AI",
    page_icon="🌿",
    layout="centered"
)

# ---------- HEADER ----------
st.title("🌿 ArvyaX AI Emotional Guide")
st.caption("Understand → Decide → Guide")

st.divider()

# ---------- INPUT ----------
st.subheader("📝 Your Reflection")

journal = st.text_area(
    "How are you feeling?",
    placeholder="Write your thoughts here..."
)

col1, col2 = st.columns(2)

with col1:
    sleep = st.slider("😴 Sleep Hours", 0.0, 10.0, 6.0)
    energy = st.slider("⚡ Energy Level", 1, 5, 3)

with col2:
    stress = st.slider("😰 Stress Level", 1, 5, 3)
    time = st.selectbox("⏰ Time of Day", ["morning", "afternoon", "evening", "night"])

st.divider()

# ---------- BUTTON ----------
analyze = st.button("🚀 Analyze", use_container_width=True)

# ---------- OUTPUT ----------
if analyze:

    if journal.strip() == "":
        st.warning("Please enter your reflection")
    else:
        data = {
            "journal_text": journal,
            "sleep_hours": sleep,
            "energy_level": energy,
            "stress_level": stress,
            "time_of_day": time,
            "ambience_type": "unknown",
            "duration_min": 10,
            "previous_day_mood": "neutral",
            "face_emotion_hint": "neutral",
            "reflection_quality": "medium"
        }

        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=data)

            if response.status_code == 200:
                result = response.json()

                st.divider()
                st.subheader("🧠 Analysis Result")

                # Emotion + Confidence
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Emotion", result["predicted_state"].capitalize())

                with col2:
                    st.metric("Confidence", f"{result['confidence']:.2f}")

                # Intensity
                st.subheader("🔥 Intensity Level")
                st.progress(int(result["predicted_intensity"]) / 5)

                # Recommendation
                st.subheader("💡 Recommendation")
                st.success(result["what_to_do"].replace("_", " ").capitalize())

                st.subheader("⏰ When to do it")
                st.info(result["when_to_do"].replace("_", " ").capitalize())

                # AI Message
                st.subheader("💬 AI Guidance")
                st.write(
                    f"You seem **{result['predicted_state']}** right now. "
                    f"Try **{result['what_to_do'].replace('_',' ')}**."
                )

            else:
                st.error("API error. Check FastAPI server.")

        except:
            st.error("Cannot connect to API. Make sure FastAPI is running.")