Interactive Data Dashboard (ML Frontend)
🎯 Overview
This project serves as the final client-facing interface for the entire portfolio. It is a visually clean and highly interactive Streamlit dashboard designed to test and showcase the prediction capabilities of the complex, production-ready MLOps API .

This project validates the full MLOps pipeline by demonstrating that the structured backend service is fully integrated and usable by non-technical stakeholders.

⚙️ Technologies & Integration
Component	Technology	Role
Frontend UI	Streamlit	Provides the user-friendly, responsive interface for data entry and result display.
API Connector	Python requests	Sends user input data (from the form) as a JSON request to the running FastAPI service.
Visualization	Custom CSS	Implements the clear, color-coded visual feedback (Green, Yellow, Red card) for risk segmentation.
Backend	FastAPI	Runs in a separate terminal to receive requests and execute the Two-Stage Semi-Supervised Pipeline.

Export to Sheets
🚀 Quick Start (Full Integrated Demo)
You must run the backend and the frontend simultaneously to see the full pipeline in action.

1. Start the API Server
Open your MLops Project folder's terminal and start the prediction service.

Bash

# In the MLops Project directory (where predict_api.py is)
uvicorn predict_api:app --reload
2. Start the Dashboard (MLops Dashboard)
Open your MLops Dashboard folder's terminal and run the dashboard.

Bash

# In the MLops Dashboard directory (where dashboard_app.py is)
streamlit run dashboard_app.py
3. Test the Prediction
Access the Streamlit dashboard in your browser, enter data (e.g., set Age=25, Duration=6), and click Predict.

The dashboard will send the request to the running API, and the result will appear as a color-coded card.

🗂️ Dependencies (requirements.txt content)
Plaintext

streamlit
pandas
requests