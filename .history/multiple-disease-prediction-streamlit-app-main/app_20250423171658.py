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

elif selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Emergency Assessment")

    # Emergency symptom scoring
    EMERGENCY_SCORES = {
        "None": 0,
        "Mild": 3,
        "Moderate": 7,
        "Severe": 12  # Higher weights for severe symptoms
    }

    # Input fields with all severity options
    st.subheader("Core Motor Symptoms")
    col1, col2 = st.columns(2)
    with col1:
        tremor = st.selectbox("Resting tremor", options=list(EMERGENCY_SCORES.keys()), index=2)  # Moderate default
        rigidity = st.selectbox("Muscle stiffness", options=list(EMERGENCY_SCORES.keys()), index=3)  # Severe default
    with col2:
        bradykinesia = st.selectbox("Slowness of movement", options=list(EMERGENCY_SCORES.keys()), index=3)  # Severe default
        postural = st.selectbox("Balance problems", options=list(EMERGENCY_SCORES.keys()), index=3)  # Severe default
    
    st.subheader("Other Symptoms")
    col1, col2 = st.columns(2)
    with col1:
        gait = st.selectbox("Walking difficulties", options=list(EMERGENCY_SCORES.keys()), index=3)  # Severe default
        speech = st.selectbox("Speech changes", options=list(EMERGENCY_SCORES.keys()), index=3)  # Severe default
    with col2:
        facial = st.selectbox("Reduced facial expression", options=list(EMERGENCY_SCORES.keys()), index=3)  # Severe default
        handwriting = st.selectbox("Small handwriting", options=list(EMERGENCY_SCORES.keys()), index=3)  # Severe default
    
    fatigue = st.selectbox("Fatigue level", options=list(EMERGENCY_SCORES.keys()), index=3)  # Severe default

    if st.button("Emergency Risk Assessment"):
        # Calculate emergency score
        emergency_score = (
            EMERGENCY_SCORES[tremor] +
            EMERGENCY_SCORES[rigidity] +
            EMERGENCY_SCORES[bradykinesia] +
            EMERGENCY_SCORES[postural] +
            EMERGENCY_SCORES[gait] +
            EMERGENCY_SCORES[speech] +
            EMERGENCY_SCORES[facial] +
            EMERGENCY_SCORES[handwriting] +
            EMERGENCY_SCORES[fatigue]
        )

        # Create model input (22 features)
        user_input = [
            # Core motor features (weighted heavily)
            EMERGENCY_SCORES[tremor] * 1.2,
            EMERGENCY_SCORES[bradykinesia] * 1.5,  # Most important cardinal feature
            EMERGENCY_SCORES[rigidity] * 1.3,
            EMERGENCY_SCORES[postural] * 1.4,
            EMERGENCY_SCORES[gait] * 1.3,
            
            # Speech and facial
            EMERGENCY_SCORES[speech] * 1.1,
            EMERGENCY_SCORES[facial] * 1.1,
            
            # Handwriting
            EMERGENCY_SCORES[handwriting] * 1.2,
            
            # Composite scores
            emergency_score,
            emergency_score / 2,  # Normalized
            emergency_score / 3,  # Weighted
            
            # Biomarker placeholders (abnormal values for severe case)
            0.82,   # UPDRS-III (severe range)
            115.0,  # Vocal frequency
            0.007,  # Jitter (abnormal)
            0.032,  # Shimmer (abnormal)
            3.1,    # NHR (abnormal)
            0.28,   # HNR (abnormal)
            158.0,  # RPDE (abnormal)
            0.72,   # DFA (abnormal)
            1.75,   # PPE (abnormal)
            0.93,   # Spread1
            1.08    # Spread2
        ]

        # Get base prediction
        proba = parkinsons_model.predict_proba([user_input])[0][1] * 100
        
        # EMERGENCY OVERRIDE SYSTEM
        if emergency_score >= 80:  # Multiple severe symptoms
            final_proba = min(max(proba * 2.5, 85), 99)  # 85-99% range
            st.error(f"🚨 PARKINSON'S EMERGENCY ({final_proba:.0f}% probability)")
            st.error("""
            IMMEDIATE ACTION REQUIRED:
            1. Urgent movement disorder specialist referral TODAY
            2. STAT DaTscan SPECT imaging
            3. Levodopa challenge test
            4. Discontinue any dopamine-blocking medications""")
        elif emergency_score >= 60:
            final_proba = min(max(proba * 2.0, 75), 90)  # 75-90% range
            st.error(f"⚠️ ADVANCED PARKINSON'S LIKELY ({final_proba:.0f}% probability)")
            st.warning("""
            URGENT RECOMMENDATIONS:
            - Neurology consult within 24-48 hours
            - Complete UPDRS assessment
            - Consider starting dopaminergic therapy""")
        else:
            final_proba = proba
            st.info(f"Parkinson's risk: {final_proba:.0f}%")








