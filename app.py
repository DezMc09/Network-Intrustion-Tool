import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

probs = model.predict_proba(df)
confidence = np.max(probs, axis=1)

df["Confidence"] = confidence

st.subheader("Predictions with Confidence")
st.dataframe(df.head())
# Load files
with open("model_small.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.set_page_config(page_title="Network Intrusion Detection Tool", layout="wide")

st.title("🔐 Network Intrusion Detection Tool")
st.write("Detect cyber attacks in network traffic using a trained machine learning model.")

input_method = st.radio("Choose Input Method", ["CSV Upload", "Manual Entry"])

def get_risk_level(conf):
    if conf >= 0.90:
        return "High"
    elif conf >= 0.70:
        return "Medium"
    return "Low"

if input_method == "CSV Upload":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview of Data")
        st.dataframe(df.head())

        missing_cols = [col for col in feature_columns if col not in df.columns]

        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            df_features = df[feature_columns].copy()

            predictions = model.predict(df_features)
            labels = label_encoder.inverse_transform(predictions)

            results_df = df.copy()
            results_df["Prediction"] = labels

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df_features)
                results_df["Confidence"] = probs.max(axis=1)
                results_df["Risk_Level"] = results_df["Confidence"].apply(get_risk_level)

            st.subheader("Predictions")
            st.dataframe(results_df.head(20))

            total_records = len(results_df)
            suspicious = (results_df["Prediction"].str.lower() != "benign").sum()

            col1, col2 = st.columns(2)
            col1.metric("Total Records", total_records)
            col2.metric("Suspicious Records", suspicious)

            st.subheader("Attack Distribution")
            counts = results_df["Prediction"].value_counts().head(15)

            fig, ax = plt.subplots(figsize=(10, 5))
            counts.plot(kind="bar", ax=ax)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "results.csv", "text/csv")
y_pred = model.predict(df)

# Convert numbers back to attack names
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Add to dataframe
df["Predicted_Label"] = y_pred_labels

st.subheader("Predictions with Labels")
st.dataframe(df.head())
elif input_method == "Manual Entry":
    st.subheader("Manual Feature Entry")
    st.write("Enter a few feature values. Any feature not shown will default to 0.")

    # Beginner-friendly manual fields
    flow_duration = st.number_input("flow_duration", min_value=0.0, value=0.0)
    header_length = st.number_input("Header_Length", min_value=0.0, value=0.0)
    protocol_type = st.number_input("Protocol Type", min_value=0.0, value=0.0)
    rate = st.number_input("Rate", min_value=0.0, value=0.0)
    syn_flag_number = st.number_input("syn_flag_number", min_value=0.0, value=0.0)
    ack_count = st.number_input("ack_count", min_value=0.0, value=0.0)
    tot_sum = st.number_input("Tot sum", min_value=0.0, value=0.0)
    iat = st.number_input("IAT", min_value=0.0, value=0.0)
    magnitude = st.number_input("Magnitude", min_value=0.0, value=0.0)
    weight = st.number_input("Weight", min_value=0.0, value=0.0)

    if st.button("Predict"):
        input_data = {col: 0 for col in feature_columns}

        # Only fill columns that exist in your trained feature list
        if "flow_duration" in input_data:
            input_data["flow_duration"] = flow_duration
        if "Header_Length" in input_data:
            input_data["Header_Length"] = header_length
        if "Protocol Type" in input_data:
            input_data["Protocol Type"] = protocol_type
        if "Rate" in input_data:
            input_data["Rate"] = rate
        if "syn_flag_number" in input_data:
            input_data["syn_flag_number"] = syn_flag_number
        if "ack_count" in input_data:
            input_data["ack_count"] = ack_count
        if "Tot sum" in input_data:
            input_data["Tot sum"] = tot_sum
        if "IAT" in input_data:
            input_data["IAT"] = iat
        if "Magnitude" in input_data:
            input_data["Magnitude"] = magnitude
        if "Weight" in input_data:
            input_data["Weight"] = weight

        input_df = pd.DataFrame([input_data])[feature_columns]

        prediction = model.predict(input_df)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]

        st.success(f"Prediction: {predicted_label}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            confidence = probs.max()
            risk = get_risk_level(confidence)

            st.write(f"Confidence: {confidence:.2%}")
            st.write(f"Risk Level: {risk}")

            prob_df = pd.DataFrame({
                "Class": label_encoder.classes_,
                "Probability": probs
            }).sort_values("Probability", ascending=False).head(10)

            st.subheader("Top Class Probabilities")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(prob_df["Class"], prob_df["Probability"])
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
