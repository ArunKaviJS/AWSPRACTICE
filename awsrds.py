import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import xgboost
# Load models locally
xgb_boost = joblib.load("xgboost_riskscore_insurance.pkl")
model_claim = joblib.load("logisticforfraud_claim_insuranceclaim.pkl")
labencode = joblib.load("Encoder_riskscore_insurance.pkl")
oneh_decoder = joblib.load("onehendoderforinsurancerisk.pkl")

# Streamlit UI
tab1, tab2 = st.tabs(["RiskCore", "Claimamount"])

with tab1:
    st.markdown(
        """
        <style>
            .gradient-text {
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                background: linear-gradient(to right, #0047AB, #007BFF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                padding: 10px;
            }
        </style>
        <div class='gradient-text'>PREDICTING RISK SCORE AND FRAUDULENT CLAIM[CLASSIFICATION]</div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        vpa = st.number_input("Vehicle Property Age", format="%.0f")
        claim_amount = st.number_input("Claim Amount", format="%.4f")
    with c2:
        claim_history = st.number_input("Claim History", format="%.0f")
        prem_amount = st.number_input("Premium Amount", format="%.0f")
    with c3:
        user_data = {
            "Policy_Type": st.text_input("Policy Type").title().strip(),
            "Gender": st.text_input("Gender").title().strip(),
        }

    if st.button("Predict"):
        user_df = pd.DataFrame([user_data])
        encoded_user_input = oneh_decoder.transform(user_df)
        num_inputs = np.array([vpa, claim_history, prem_amount, claim_amount]).reshape(1, -1)
        combined_input = np.concatenate([num_inputs, encoded_user_input], axis=1)
        riskscore = xgb_boost.predict(combined_input)[0]
        fraudulentclaim = model_claim.predict(combined_input)[0]

        with st.expander("View User Inputs"):
            st.write(f"Vehicle Age: {vpa}")
            st.write(f"Claim History: {claim_history}")
            st.write(f"Premium Amount: {prem_amount}")
            st.write(f"Claim Amount: {claim_amount}")
            st.write(f"Encoded Data: {encoded_user_input}")
            st.write(f"Risk Score: {riskscore}")
            st.write(f"Fraudulent Claim: {fraudulentclaim}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h2 style='text-align: center;'>RISK LEVEL</h2>", unsafe_allow_html=True)
            if riskscore == 1:
                st.image("https://th.bing.com/th/id/OIP.mjdRpGUbjZYMAExB77HkmAHaHa?w=600&h=600&rs=1&pid=ImgDetMain")
                st.success("Low")
            elif riskscore == 2:
                st.image("https://th.bing.com/th/id/OIP.qLtHrKM0j_WnBop_ZdOgHgHaHa?rs=1&pid=ImgDetMain")
                st.warning("Medium")
            else:
                st.image("https://th.bing.com/th/id/OIP.PjxfjvSKrty6DpRmD0AoYAHaHa?w=600&h=600&rs=1&pid=ImgDetMain")
                st.error("High")

        with col2:
            st.markdown("<h2 style='text-align: center;'>FRAUDULENT CLAIM</h2>", unsafe_allow_html=True)
            if fraudulentclaim == 0:
                st.image("https://cdn3.iconfinder.com/data/icons/users-48/112/01-users_user-approve-check-512.png")
                st.success("NOT A FRAUD")
            else:
                st.image("https://img.freepik.com/premium-vector/red-triangle-with-word-fraud-alert-it_691833-169.jpg")
                st.error("FRAUD")

with tab2:
    st.markdown(
        """
        <style>
            .gradient-text {
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                background: linear-gradient(to right, #0047AB, #007BFF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                padding: 10px;
            }
        </style>
        <div class='gradient-text'>CLAIM AMOUNT [REGRESSION]</div>
        """,
        unsafe_allow_html=True,
    )