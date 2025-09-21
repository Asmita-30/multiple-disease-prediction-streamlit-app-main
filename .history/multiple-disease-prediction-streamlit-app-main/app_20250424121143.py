import os
import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu  # Make sure this is properly installed

# Set page configuration
st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑⚕️"
)

# Get working directory
working_dir = os.path.dirname(os.path.abspath(__file__))


# loading the saved models
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
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML - Emergency Edition')

    # Emergency symptom scoring
    SYMPTOM_WEIGHTS = {
        'Excessive thirst': 8,       # Polydipsia
        'Excessive hunger': 7,       # Polyphagia
        'Blurry vision': 9,          # Retinopathy risk
        'Numbness/tingling': 6,      # Neuropathy
        'Dry mouth': 4,
        'Slow wound healing': 10,     # Strongest predictor
        'Fatigue': 5
    }

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1: 
        Gender = st.selectbox('Gender', ['Male', 'Female'])
        Age = st.number_input('Age', 1, 120, 33)
        Thirst = st.selectbox('Thirst', ['Normal', 'Excessive thirst'])
    with col2:
        Hunger = st.selectbox('Hunger', ['Normal', 'Excessive hunger'])
        Vision = st.selectbox('Vision', ['Normal', 'Blurry vision'])
        Tingling = st.selectbox('Tingling', ['None', 'Numbness/tingling'])
    with col3:
        DryMouth = st.selectbox('Dry Mouth', ['No', 'Dry mouth'])
        WoundHealing = st.selectbox('Wound Healing', ['Normal', 'Slow wound healing'])
        Fatigue = st.selectbox('Fatigue', ['No', 'Fatigue'])

    if st.button('Assess Diabetes Risk'):
        # Calculate emergency score (0-100)
        emergency_score = 0
        if Thirst == 'Excessive thirst': emergency_score += SYMPTOM_WEIGHTS['Excessive thirst']
        if Hunger == 'Excessive hunger': emergency_score += SYMPTOM_WEIGHTS['Excessive hunger']
        if Vision == 'Blurry vision': emergency_score += SYMPTOM_WEIGHTS['Blurry vision']
        if Tingling == 'Numbness/tingling': emergency_score += SYMPTOM_WEIGHTS['Numbness/tingling']
        if DryMouth == 'Dry mouth': emergency_score += SYMPTOM_WEIGHTS['Dry mouth']
        if WoundHealing == 'Slow wound healing': emergency_score += SYMPTOM_WEIGHTS['Slow wound healing']
        if Fatigue == 'Fatigue': emergency_score += SYMPTOM_WEIGHTS['Fatigue']

        # Age adjustment
        age_risk = min(Age/2, 30)
        total_risk = emergency_score + age_risk

        # Model prediction with clinical override
        user_input = [
            1 if Gender == 'Male' else 0,
            Age,
            1 if Thirst == 'Excessive thirst' else 0,
            1 if Hunger == 'Excessive hunger' else 0,
            1 if Vision == 'Blurry vision' else 0,
            1 if WoundHealing == 'Slow wound healing' else 0,
            total_risk/10,  # Normalized risk score
            1 if emergency_score > 15 else 0  # Emergency flag
        ]

        # Get base prediction
        diab_proba = diabetes_model.predict_proba([user_input])[0][1] * 100
        
        # EMERGENCY OVERRIDE SYSTEM
        if emergency_score >= 20:  # Multiple concerning symptoms
            final_proba = max(diab_proba * 3, 70)  # Minimum 70% if serious symptoms
            st.error(f"🚨 DIABETES EMERGENCY ({final_proba:.0f}% probability)")
            st.error("""
            CRITICAL FINDINGS:
            - Multiple diabetic emergency symptoms detected
            - Possible hyperglycemic crisis
            IMMEDIATE ACTIONS REQUIRED:
            1. Emergency blood glucose test
            2. Urgent HbA1c and ketones test
            3. IV fluids if glucose > 300 mg/dL
            4. Endocrine consult STAT""")
        elif emergency_score >= 15:
            final_proba = max(diab_proba * 2, 50)  # Minimum 50%
            st.error(f"RISK DIABETES ({final_proba:.0f}% probability)")
            st.warning("""
            URGENT RECOMMENDATIONS:
            - Same-day glucose testing
            - HbA1c within 24 hours
            - Ophthalmology referral""")
        else:
            final_proba = diab_proba
            st.info(f"Diabetes probability: {final_proba:.0f}%")



elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Risk Assessment')

    # Symptom severity scoring
    SYMPTOM_SCORES = {
        'No chest pain': 0,
        'Mild chest discomfort': 3,
        'Occasional chest pain': 7,
        'Pain during activity': 10,
        'Low': 1,
        'Normal': 0,
        'High': 3,
        'No': 0,
        'Yes': 5
    }

    # Input fields matching your exact UI
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Your age', min_value=1, max_value=120, value=54)
        sex = st.selectbox('Gender', ['Male', 'Female'])
        chest_pain = st.selectbox('Do you feel chest pain?', 
                                ['No chest pain', 'Mild chest discomfort', 
                                 'Occasional chest pain', 'Pain during activity'])
    with col2:
        blood_pressure = st.selectbox('Usual blood pressure', ['Low', 'Normal', 'High'])
        heart_rate = st.selectbox('Usual heart rate', ['Low', 'Normal', 'High'])
        exercise_pain = st.selectbox('Pain during exercise?', ['No', 'Yes'])
    with col3:
        tiredness = st.selectbox('Do you get tired easily?', ['No', 'Yes'])
        breathless = st.selectbox('Do you feel breathless?', ['No', 'Yes'])
        swelling = st.selectbox('Swelling in feet or ankles?', ['No', 'Yes'])
    
    # Additional symptoms in separate rows
    dizziness = st.selectbox('Do you feel dizzy or faint?', ['No', 'Yes'])
    heartbeat = st.selectbox('Irregular heartbeat?', ['No', 'Yes'])
    nausea = st.selectbox('Nausea or sweating without reason?', ['No', 'Yes'])
    fasting_bs = st.selectbox('Fasting blood sugar > 120 mg/dl?', ['No', 'Yes'])

    if st.button('Heart Disease Test Result'):
        # Calculate risk score
        risk_score = (
            SYMPTOM_SCORES[chest_pain] +
            SYMPTOM_SCORES[heart_rate] +
            SYMPTOM_SCORES[exercise_pain] +
            SYMPTOM_SCORES[tiredness] +
            SYMPTOM_SCORES[breathless] +
            SYMPTOM_SCORES[swelling] +
            SYMPTOM_SCORES[dizziness] +
            SYMPTOM_SCORES[heartbeat] +
            SYMPTOM_SCORES[nausea]
        )
        
        # Create model input (13 features)
        user_input = [
            float(age),
            1 if sex == 'Male' else 0,
            SYMPTOM_SCORES[chest_pain],
            120 if blood_pressure == 'Normal' else 140 if blood_pressure == 'High' else 100,
            200,  # cholesterol placeholder
            1 if fasting_bs == 'Yes' else 0,
            1 if heartbeat == 'Yes' else 0,  # ECG
            SYMPTOM_SCORES[heart_rate] * 20 + 60,  # Convert to bpm
            SYMPTOM_SCORES[exercise_pain] > 0,
            2.0 if exercise_pain == 'Yes' else 0.5,  # ST depression
            1 if breathless == 'Yes' else 0,  # exang
            2 if risk_score > 20 else 1,  # slope
            1 if swelling == 'Yes' else 0  # vessels
        ]

        # Verify correct number of features
        if len(user_input) != 13:
            st.error("System error: Incorrect feature count")
        else:
            # Get prediction
            heart_proba = heart_disease_model.predict_proba([user_input])[0][1] * 100
            
            # Clinical risk adjustment
            if risk_score > 25:
                adjusted_proba = min(heart_proba * 2.5, 95)
                st.error(f"🚨 CRITICAL HEART DISEASE RISK ({adjusted_proba:.0f}% probability)")
                st.error("""
                EMERGENCY WARNING:
                - Multiple high-risk symptoms detected
                - Possible acute coronary syndrome
                IMMEDIATE ACTIONS:
                1. Call emergency services
                2. Chew 325mg aspirin if available
                3. Remain seated while waiting for help""")
            elif risk_score > 15:
                adjusted_proba = min(heart_proba * 1.8, 80)
                st.warning(f" HEART DISEASE RISK ({adjusted_proba:.0f}% probability)")
                st.warning("""
                URGENT RECOMMENDATIONS:
                - Same-day cardiology consult
                - ECG and troponin tests
                - Blood pressure monitoring""")
            else:
                st.info(f"Heart disease risk: {heart_proba:.0f}%")

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





