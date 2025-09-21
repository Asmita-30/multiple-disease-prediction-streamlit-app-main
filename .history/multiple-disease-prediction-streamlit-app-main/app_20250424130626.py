import os
import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕")

# Getting the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))


# Loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                         ['Diabetes Prediction',
                          'Heart Disease Prediction',
                          "Parkinson's Prediction"],
                         menu_icon='hospital-fill',
                         icons=['activity', 'heart', 'person'],
                         default_index=0)

# -------------------- Diabetes Prediction --------------------
# -------------------- Diabetes Prediction --------------------
if selected == 'Diabetes Prediction':
    st.title('Diabetes Risk Check')

    st.subheader("🧑 About You")
    gender = st.radio('What is your gender?', ['Male', 'Female'], key='diab_gender')
    if gender == 'Female':
        pregnancies = st.radio('Have you been pregnant before?', ['No', 'Yes, once', 'Yes, multiple times'], key='diab_pregnancy')
        preg_score = ['No', 'Yes, once', 'Yes, multiple times'].index(pregnancies)
    else:
        preg_score = 0

    st.subheader("🩺 Your Symptoms")
    thirst = st.radio('Do you feel unusually thirsty?', ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
    hunger = st.radio('Do you feel hungry even after meals?', ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
    vision = st.radio('Is your vision blurry?', ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
    wounds = st.radio('Do cuts or wounds take longer to heal?', ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
    fatigue = st.radio('Do you feel tired often?', ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
    tingling = st.radio('Do you feel tingling in your hands or feet?', ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])

    symptom_scores = [thirst, hunger, vision, wounds, fatigue, tingling]
    weights = [8, 7, 9, 10, 5, 6]
    total_score = sum(weights[i] * ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'].index(symptom_scores[i]) for i in range(len(weights)))
    total_score += preg_score * 5

    if total_score >= 75:
        st.error(f"🚨 High Diabetes Risk: {min(100, total_score)}%")
    elif total_score >= 50:
        st.warning(f"⚠ Moderate Risk: {min(100, total_score)}%")
    elif total_score >= 25:
        st.info(f"🟡 Mild Risk: {min(100, total_score)}%")
    else:
        st.success(f"✅ Low Risk: {min(100, total_score)}%")

# -------------------- Heart Disease Prediction --------------------
elif selected == 'Heart Disease Prediction':
    st.title('Heart Risk Check')

    st.subheader("❤ Your Symptoms")
    chest_pain = st.radio('Do you feel chest tightness or pain?', ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
    tired = st.radio('Do you feel tired doing small activities?', ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
    short_breath = st.radio('Do you feel breathless often?', ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
    sweat = st.radio('Do you sweat more than usual while resting?', ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
    dizziness = st.radio('Do you feel dizzy or faint?', ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
    heartbeat = st.radio('Is your heartbeat irregular or very fast?', ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])

    symptom_levels = [chest_pain, tired, short_breath, sweat, dizziness, heartbeat]
    severity_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4}
    weights = [10, 8, 9, 7, 6, 8]

    score = sum(severity_mapping[level] * weights[i] for i, level in enumerate(symptom_levels))
    max_possible_score = sum(w * 4 for w in weights)  # Max if all "Always"
    percentage = round((score / max_possible_score) * 100)

    if percentage >= 75:
        st.error(f"🔴 High Heart Disease Risk: {percentage}%")
    elif percentage >= 50:
        st.warning(f"🟠 Moderate Risk: {percentage}%")
    elif percentage >= 25:
        st.info(f"🟡 Mild Risk: {percentage}%")
    else:
        st.success(f"🟢 Low Risk: {percentage}%")

# -------------------- Parkinson's Prediction --------------------
# Parkinson's Prediction
elif selected == "Parkinson's Prediction":
    st.title("🧠 Parkinson's Voice Screening")
    st.write("Complete this quick voice assessment to check your risk level.")

    # Symptom Questions
    st.subheader("🗣 Voice & Speech")
    col1, col2 = st.columns(2)
    with col1:
        voice_pitch = st.select_slider("Speaking pitch", options=["Very high", "High", "Normal", "Low", "Very low"])
        voice_shaking = st.select_slider("Voice shaking", options=["Never", "Rarely", "Sometimes", "Often", "Always"])
    with col2:
        voice_volume = st.select_slider("Volume fluctuation", options=["Never", "Rarely", "Sometimes", "Often", "Always"])
        voice_roughness = st.select_slider("Voice roughness", options=["Never", "Rarely", "Sometimes", "Often", "Always"])

    st.subheader("💬 Speech Clarity")
    clarity = st.select_slider("Speech understanding difficulty", options=["Never", "Rarely", "Sometimes", "Often", "Always"])

    st.subheader("⏳ Symptom History")
    duration = st.select_slider("Duration of symptoms", options=["No changes", "Less than 6 months", "6-12 months", "1-2 years", "Over 2 years"])
    severity = st.select_slider("Severity of symptoms", options=["Not at all", "Mild", "Moderate", "Severe", "Very severe"])

    # Dynamic score calculation
    def compute_risk():
        mapping = lambda x, options: options.index(x)
        score = (
            mapping(voice_pitch, ["Very high", "High", "Normal", "Low", "Very low"]) * 4 +
            mapping(voice_shaking, ["Never", "Rarely", "Sometimes", "Often", "Always"]) * 8 +
            mapping(voice_volume, ["Never", "Rarely", "Sometimes", "Often", "Always"]) * 6 +
            mapping(voice_roughness, ["Never", "Rarely", "Sometimes", "Often", "Always"]) * 6 +
            mapping(clarity, ["Never", "Rarely", "Sometimes", "Often", "Always"]) * 10 +
            mapping(duration, ["No changes", "Less than 6 months", "6-12 months", "1-2 years", "Over 2 years"]) * 4 +
            mapping(severity, ["Not at all", "Mild", "Moderate", "Severe", "Very severe"]) * 12
        )

        # Scaling to 0–100
        scaled = min(100, (score / 150) * 100)

        # Mild boost if severity & clarity are bad
        if mapping(severity, ["Not at all", "Mild", "Moderate", "Severe", "Very severe"]) >= 3 and \
           mapping(clarity, ["Never", "Rarely", "Sometimes", "Often", "Always"]) >= 3:
            scaled *= 1.2

        return min(scaled, 100)

    risk = compute_risk()

    st.subheader("📊 Estimated Risk")
    if risk >= 75:
        st.error(f"🔴 High Risk: {risk:.1f}%")
        st.write("Several strong indicators detected. Please consult a neurologist.")
    elif risk >= 50:
        st.warning(f"🟠 Moderate Risk: {risk:.1f}%")
        st.write("Some signs present. Keep monitoring and consider getting checked.")
    elif risk >= 25:
        st.info(f"🟡 Mild Risk: {risk:.1f}%")
        st.write("Minor symptoms detected. Stay observant.")
    else:
        st.success(f"🟢 Low Risk: {risk:.1f}%")
        st.write("No major symptoms detected.")

    st.progress(risk / 100)