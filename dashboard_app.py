# Goal: Create a friendly Streamlit frontend to communicate with the complex MLOps Prediction API (Project 2).
# This file collects user input, sends it via HTTP request, and visualizes the prediction.

import streamlit as st
import requests  # Essential for making HTTP requests to the FastAPI backend
import pandas as pd  # Used for structuring the input data
import json
import numpy as np

# --- 1. CONFIGURATION ---
# IMPORTANT: This must match the running port of your FastAPI server (Project 2)
API_URL = "http://127.0.0.1:8000/predict_segment/"

# Define the expected categorical feature options based on the German Credit Data
JOB_OPTIONS = {0: 'Unskilled-Non-Resident', 1: 'Unskilled-Resident', 2: 'Skilled', 3: 'Highly Skilled'}


# --- 2. CORE PREDICTION LOGIC ---

def get_prediction(data):
    """Sends the structured data to the FastAPI endpoint and returns the prediction."""
    try:
        # The data must be sent as JSON that matches the Pydantic schema in predict_api.py
        response = requests.post(API_URL, json=data)

        # Check for successful response
        if response.status_code == 200:
            return response.json()
        else:
            # Handle API errors (e.g., validation failure, server error)
            error_detail = response.json().get("detail", "Unknown API Error")
            st.error(f"API Error ({response.status_code}): {error_detail}")
            st.warning("Ensure your Project 2 FastAPI server is running in a separate terminal.")
            return None

    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the Prediction API.")
        st.warning(f"Please ensure your Project 2 FastAPI server is running at {API_URL}.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# --- 3. UI SETUP AND INPUT FORM ---

st.set_page_config(page_title="Risk Segmentation Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Modern Dashboard Styling */
    :root {
        --bg-primary: #121212;
        --bg-secondary: #1e1e1e;
        --text-color: #e0e0e0;
        --accent-color: #1abc9c; /* Teal for high-impact buttons/results */
        --low-risk: #27ae60;    /* Green */
        --medium-risk: #f1c40f; /* Yellow */
        --high-risk: #e74c3c;   /* Red */
        --border-color: #333333;
    }

    .stApp { background-color: var(--bg-primary); color: var(--text-color); }

    .stTitle { 
        color: var(--accent-color); 
        font-weight: 700;
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 10px;
    }

    /* Input containers */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input {
        background-color: var(--bg-secondary) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 10px;
    }

    /* Selectbox Fix: Ensure the selected text inside the box is clearly visible */
    div.stSelectbox > div[data-testid="stTextInput"], 
    div.stSelectbox > div[role="combobox"] div[data-testid="stTextInput"] {
        background-color: var(--bg-secondary) !important;
        color: var(--text-color) !important; 
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    /* Ensure the selected value text is also correctly colored */
    div.stSelectbox div[data-baseweb="select"] div[role="button"] {
        color: var(--text-color) !important;
    }


    /* Custom Result Card Styles */
    .risk-card {
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        color: white;
        text-align: center;
        margin-top: 30px;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .risk-label {
        font-size: 2.5em;
        font-weight: 800;
    }
    .risk-segment {
        font-size: 1.2em;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Credit Risk Segmentation Dashboard")
st.caption("Frontend for the Two-Stage MLOps Prediction API (Project 2).")

# --- Form Columns ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal & Financial")
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=35)
    credit_amount = st.number_input("Credit Amount (DM)", min_value=500, value=6500)
    duration = st.number_input("Loan Duration (Months)", min_value=6, max_value=72, value=24)

with col2:
    st.subheader("Employment & Housing")
    # FIX 1: Removed external st.markdown confirmation
    sex = st.selectbox("Sex", options=['male', 'female'], index=0)

    job_index = st.selectbox("Job Classification", options=list(JOB_OPTIONS.keys()),
                             format_func=lambda x: JOB_OPTIONS[x])
    # FIX 2: Removed external st.markdown confirmation

    housing = st.selectbox("Housing Status", options=['rent', 'own', 'free'], index=1)
    # FIX 3: Removed external st.markdown confirmation

with col3:
    st.subheader("Account Status")
    saving_accounts = st.selectbox("Saving Accounts", options=['little', 'moderate', 'quite rich', 'rich'], index=0)
    # FIX 4: Removed external st.markdown confirmation

    checking_account = st.selectbox("Checking Account", options=['little', 'moderate', 'rich', 'no checking'], index=1)
    # FIX 5: Removed external st.markdown confirmation

    purpose = st.selectbox("Purpose of Loan",
                           options=['car', 'furniture/equipment', 'radio/TV', 'domestic appliances', 'repairs',
                                    'education', 'business', 'vacation/others'], index=0)
    # FIX 6: Removed external st.markdown confirmation

st.markdown("---")

# --- Prediction Button ---
if st.button("Predict Risk Segment", key='predict_button'):
    # 4. Gather data for API
    input_data = {
        "Age": int(age),
        "Duration": int(duration),
        "Credit_amount": int(credit_amount),
        "Job": int(job_index),
        "Sex": sex,
        "Housing": housing,
        "Saving_accounts": saving_accounts,
        "Checking_account": checking_account,
        "Purpose": purpose,
    }

    # 5. Get Prediction from API
    with st.spinner("Analyzing data via MLOps Pipeline..."):
        prediction_result = get_prediction(input_data)

    # 6. Display Result
    if prediction_result:
        segment_id = prediction_result.get("risk_segment_id", 99)
        segment_name = prediction_result.get("risk_segment_name", "Unknown Segment")

        # --- Visual Mapping for segments ---
        # 0 = Low Risk (Green), 1 = Medium Risk (Yellow), 2 = High Risk (Red)
        if segment_id == 0:
            color = "var(--low-risk)"
            label = "Low Risk Segment"
        elif segment_id == 1:
            color = "var(--medium-risk)"
            label = "Medium Risk Segment"
        elif segment_id == 2:
            color = "var(--high-risk)"
            label = "High Risk Segment"
        else:
            color = "var(--border-color)"
            label = "Prediction Error"

        st.markdown(f"""
        <div class="risk-card" style="background-color: {color};">
            <div class="risk-label">{label}</div>
            <div class="risk-segment">Predicted Segment: {segment_name}</div>
        </div>
        """, unsafe_allow_html=True)

        st.success("Prediction complete! Result received from FastAPI API.")

# --- Instructions for users ---
st.markdown("---")
st.info("To run this dashboard, ensure your Project 2 (FastAPI) server is active and running on port 8000.")
