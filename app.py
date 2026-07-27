import streamlit as st
from disease import predict_disease, X

st.set_page_config(page_title="Disease Prediction")

st.title("🩺 Disease Prediction System")

# Get all symptom names from the training data
symptom_list = sorted(X.columns.tolist())

# User selects symptoms
selected_symptoms = st.multiselect(
    "Select your symptoms:",
    symptom_list
)

# Predict button
if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")

    else:
        disease, description, precautions = predict_disease(selected_symptoms)

        st.success(f"Predicted Disease: **{disease}**")

        st.subheader("Disease Description")
        st.write(description)

        st.subheader("Recommended Precautions")

        for i, precaution in enumerate(precautions, start=1):
            st.write(f"{i}. {precaution}")


