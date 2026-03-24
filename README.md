🔐 SmartNIDS: Machine Learning-Based Network Intrusion Detection System
📌 Overview

SmartNIDS is a machine learning-powered network intrusion detection system designed to identify malicious activity in network traffic. This project leverages supervised learning models to classify network behavior as either benign or attack, providing real-time insights through an interactive dashboard.

🚀 Features
🔍 Detects cyber attacks in network traffic
🤖 Machine learning-based classification (benign vs attack)
📊 Confidence score for predictions
🖥️ Interactive dashboard built with Streamlit
📈 Data visualization and model output display
🧠 Technologies Used
Python
Pandas & NumPy (data processing)
Scikit-learn (machine learning models)
Matplotlib (visualization)
Streamlit (web interface)
📂 Dataset

This project uses publicly available network traffic datasets (e.g., Kaggle intrusion detection datasets), which include:

Benign traffic data
Multiple categories of cyber attacks
⚙️ Project Workflow
Data Collection
Imported benign and attack datasets
Data Preprocessing
Cleaned and merged datasets
Encoded categorical features
Selected relevant features
Feature Engineering
Standardized and aligned dataset columns
Prepared features for model training
Model Training
Split dataset into training and testing sets
Trained machine learning models for classification
Evaluation
Assessed model performance using accuracy and other metrics
Deployment
Built a Streamlit interface for real-time predictions
📊 Sample Output
Prediction: Attack / Benign
Confidence Score: e.g., 92% certainty
▶️ How to Run the Project
Clone the repository:
git clone https://github.com/yourusername/smartnids.git
cd smartnids
Install dependencies:
pip install -r requirements.txt
Run the Streamlit app:
streamlit run app.py
📁 Project Structure
smartnids/
│
├── app.py                  # Streamlit dashboard
├── model_small.pkl         # Trained ML model
├── label_encoder.pkl       # Label encoder
├── feature_columns.pkl     # Feature list
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks (optional)
├── requirements.txt        # Dependencies
└── README.md
📈 Future Improvements
Improve model accuracy with advanced algorithms
Add deep learning models (Neural Networks)
Expand detection to IoT-based attacks
Deploy as a web-based cybersecurity tool
📚 Research & Development

This project is part of ongoing research exploring the integration of machine learning and cybersecurity for intelligent threat detection systems.

👩🏽‍💻 Author

Alkendria McNair
Cybersecurity | Machine Learning | Research

📬 Feedback

Feel free to connect or provide feedback on improving this system!
