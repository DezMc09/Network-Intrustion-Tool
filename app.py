import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Network Intrusion Detection Tool", layout="wide")

st.title("🔐 Network Intrusion Detection Tool")
st.write("Detect cyber attacks in network traffic using a trained machine learning model.")

# -----------------------------
# Load model files
# -----------------------------
with open("model_small.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


# -----------------------------
# Helper functions
# -----------------------------
def get_risk_level(confidence):
    if confidence >= 0.90:
        return "High"
    elif confidence >= 0.70:
        return "Medium"
    return "Low"


def is_benign_label(label):
    label = str(label).lower()
    return "benign" in label or "normal" in label


# -----------------------------
# Input method selector
# -----------------------------
input_method = st.radio("Choose Input Method", ["CSV Upload", "Manual Entry"])


# -----------------------------
# CSV Upload Section
# -----------------------------
if input_method == "CSV Upload":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview of Data")
        st.dataframe(df.head())

        # Check for missing required columns
        missing_cols = [col for col in feature_columns if col not in df.columns]

        if missing_cols:
            st.error("Your CSV is missing required columns.")
            st.write(missing_cols)
        else:
            # Keep columns in same order used during training
            df_features = df[feature_columns].copy()

            # Predictions
            predictions = model.predict(df_features)
            predicted_labels = label_encoder.inverse_transform(predictions)

            results_df = df.copy()
            results_df["Predicted_Label"] = predicted_labels

            # Confidence + risk
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(df_features)
                max_confidence = np.max(probabilities, axis=1)

                results_df["Confidence"] = max_confidence
                results_df["Risk_Level"] = results_df["Confidence"].apply(get_risk_level)

            # Attack or benign type
            results_df["Traffic_Type"] = results_df["Predicted_Label"].apply(
                lambda x: "Benign" if is_benign_label(x) else "Attack"
            )

            st.subheader("Predictions")
            st.dataframe(results_df.head(20))

            # Metrics
            total_records = len(results_df)
            suspicious_records = (results_df["Traffic_Type"] == "Attack").sum()
            benign_records = (results_df["Traffic_Type"] == "Benign").sum()
            attack_percentage = (suspicious_records / total_records) * 100 if total_records > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", total_records)
            col2.metric("Suspicious Records", suspicious_records)
            col3.metric("Attack %", f"{attack_percentage:.2f}%")

            # Attack distribution chart
            st.subheader("Attack Distribution")
            attack_counts = results_df["Predicted_Label"].value_counts().head(15)

            fig, ax = plt.subplots(figsize=(10, 5))
            attack_counts.plot(kind="bar", ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Count")
            ax.set_title("Top Predicted Attack Types")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

            # Benign vs Attack chart
            st.subheader("Benign vs Attack Summary")
            traffic_counts = results_df["Traffic_Type"].value_counts()

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            traffic_counts.plot(kind="bar", ax=ax2)
            ax2.set_xlabel("Traffic Type")
            ax2.set_ylabel("Count")
            ax2.set_title("Benign vs Attack")
            plt.tight_layout()
            st.pyplot(fig2)

            # Download button
            csv_output = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results",
                data=csv_output,
                file_name="predicted_results.csv",
                mime="text/csv"
            )


# -----------------------------
# Manual Entry Section
# -----------------------------
elif input_method == "Manual Entry":
    st.subheader("Manual Feature Entry")
    st.write("Enter a few feature values below. Any feature not listed will be set to 0.")

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
        # Default all features to 0
        input_data = {col: 0 for col in feature_columns}

        # Fill only matching feature names
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

        st.subheader("Manual Prediction Result")
        st.success(f"Prediction: {predicted_label}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            confidence = np.max(probabilities)
            risk_level = get_risk_level(confidence)

            st.write(f"Confidence: {confidence:.2%}")
            st.write(f"Risk Level: {risk_level}")

            prob_df = pd.DataFrame({
                "Class": label_encoder.classes_,
                "Probability": probabilities
            }).sort_values("Probability", ascending=False).head(10)

            st.subheader("Top Class Probabilities")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(prob_df["Class"], prob_df["Probability"])
            ax.set_xlabel("Class")
            ax.set_ylabel("Probability")
            ax.set_title("Top Prediction Probabilities")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
