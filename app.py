import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="Network Intrusion Detection Tool",
    page_icon="🔐",
    layout="wide"
)

# -----------------------------------
# Load model files
# -----------------------------------
with open("model_small.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# -----------------------------------
# Helper functions
# -----------------------------------
def get_risk_level(confidence):
    if confidence >= 0.90:
        return "High"
    elif confidence >= 0.70:
        return "Medium"
    return "Low"


def is_benign_label(label):
    label = str(label).lower()
    return "benign" in label or "normal" in label


def make_downloadable_csv(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


# -----------------------------------
# Header
# -----------------------------------
st.title("🔐 Network Intrusion Detection Tool")
st.markdown(
    "Analyze network traffic records using a machine learning model to identify possible cyber attacks."
)

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.header("Input Options")
input_method = st.sidebar.radio("Choose Input Method", ["CSV Upload", "Manual Entry"])

st.sidebar.markdown("---")
st.sidebar.info(
    "Tip: CSV Upload is best for batch analysis. Manual Entry is useful for testing one record at a time."
)

# -----------------------------------
# CSV Upload Section
# -----------------------------------
if input_method == "CSV Upload":
    st.subheader("📂 Upload Network Traffic CSV")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head(), use_container_width=True)

        missing_cols = [col for col in feature_columns if col not in df.columns]

        if missing_cols:
            st.error("Your CSV file is missing required columns.")
            st.write("Missing columns:")
            st.write(missing_cols)
        else:
            df_features = df[feature_columns].copy()

            predictions = model.predict(df_features)
            predicted_labels = label_encoder.inverse_transform(predictions)

            results_df = df.copy()
            results_df["Predicted_Label"] = predicted_labels

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(df_features)
                max_confidence = np.max(probabilities, axis=1)
                results_df["Confidence"] = max_confidence
                results_df["Risk_Level"] = results_df["Confidence"].apply(get_risk_level)

            results_df["Traffic_Type"] = results_df["Predicted_Label"].apply(
                lambda x: "Benign" if is_benign_label(x) else "Attack"
            )

            st.subheader("Prediction Results")
            st.dataframe(results_df.head(20), use_container_width=True)

            total_records = len(results_df)
            suspicious_records = (results_df["Traffic_Type"] == "Attack").sum()
            benign_records = (results_df["Traffic_Type"] == "Benign").sum()
            attack_percentage = (suspicious_records / total_records) * 100 if total_records > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", total_records)
            col2.metric("Suspicious Records", suspicious_records)
            col3.metric("Benign Records", benign_records)
            col4.metric("Attack %", f"{attack_percentage:.2f}%")

            st.subheader("📊 Attack Distribution")
            attack_counts = results_df["Predicted_Label"].value_counts().head(15)

            fig1, ax1 = plt.subplots(figsize=(10, 5))
            attack_counts.plot(kind="bar", ax=ax1)
            ax1.set_xlabel("Predicted Label")
            ax1.set_ylabel("Count")
            ax1.set_title("Top Predicted Attack Types")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig1)

            st.subheader("📈 Benign vs Attack Summary")
            traffic_counts = results_df["Traffic_Type"].value_counts()

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            traffic_counts.plot(kind="bar", ax=ax2)
            ax2.set_xlabel("Traffic Type")
            ax2.set_ylabel("Count")
            ax2.set_title("Benign vs Attack")
            plt.tight_layout()
            st.pyplot(fig2)

            st.subheader("⬇️ Download Results")
            csv_output = make_downloadable_csv(results_df)
            st.download_button(
                label="Download Prediction Results CSV",
                data=csv_output,
                file_name="network_intrusion_results.csv",
                mime="text/csv"
            )

# -----------------------------------
# Manual Entry Section
# -----------------------------------
elif input_method == "Manual Entry":
    st.subheader("⌨️ Manual Feature Entry")
    st.write("Enter values for a few important features. All other features will default to 0.")

    col1, col2 = st.columns(2)

    with col1:
        flow_duration = st.number_input("flow_duration", min_value=0.0, value=0.0)
        header_length = st.number_input("Header_Length", min_value=0.0, value=0.0)
        protocol_type = st.number_input("Protocol Type", min_value=0.0, value=0.0)
        rate = st.number_input("Rate", min_value=0.0, value=0.0)
        syn_flag_number = st.number_input("syn_flag_number", min_value=0.0, value=0.0)

    with col2:
        ack_count = st.number_input("ack_count", min_value=0.0, value=0.0)
        tot_sum = st.number_input("Tot sum", min_value=0.0, value=0.0)
        iat = st.number_input("IAT", min_value=0.0, value=0.0)
        magnitude = st.number_input("Magnitude", min_value=0.0, value=0.0)
        weight = st.number_input("Weight", min_value=0.0, value=0.0)

    if st.button("Predict Record"):
        input_data = {col: 0 for col in feature_columns}

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

        manual_results_df = input_df.copy()
        manual_results_df["Predicted_Label"] = predicted_label

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            confidence = np.max(probabilities)
            risk_level = get_risk_level(confidence)

            manual_results_df["Confidence"] = confidence
            manual_results_df["Risk_Level"] = risk_level

            st.write(f"Confidence: {confidence:.2%}")
            st.write(f"Risk Level: {risk_level}")

            prob_df = pd.DataFrame({
                "Class": label_encoder.classes_,
                "Probability": probabilities
            }).sort_values("Probability", ascending=False).head(10)

            st.subheader("Top Class Probabilities")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.bar(prob_df["Class"], prob_df["Probability"])
            ax3.set_xlabel("Class")
            ax3.set_ylabel("Probability")
            ax3.set_title("Top Prediction Probabilities")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig3)

        st.subheader("Manual Entry Result Table")
        st.dataframe(manual_results_df, use_container_width=True)

        st.subheader("⬇️ Download Manual Result")
        manual_csv = make_downloadable_csv(manual_results_df)
        st.download_button(
            label="Download Manual Prediction CSV",
            data=manual_csv,
            file_name="manual_prediction_result.csv",
            mime="text/csv"
        )
