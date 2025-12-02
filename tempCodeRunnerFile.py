import streamlit as st
import numpy as np
import joblib

# Load models
scaler = joblib.load("scaler.pkl")
multi = joblib.load("catboost_softlabel.pkl")
rfc = joblib.load("rf_classifier.pkl")

features = [
    'Traditional Family', 'Extraversion', 'Neuroticism', 'Agreeableness', 
    'Concientiuosness', 'Openness', 'Factor A','Factor B','Factor C','Factor E',
    'Factor F','Factor G','Factor H','Factor I','Factor L','Factor M','Factor N',
    'Factor O','Factor Q1','Factor Q2','Factor Q3','Factor Q4',
    'E','I','S','N','F','T','J','P'
]

def temperature_softmax(logits, T=1.5):
    logits = np.clip(logits, -20, 20)
    exp_vals = np.exp(logits / T)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("🧠 Personality Cluster Prediction System")
st.write("Enter the 30 questionnaire values below:")

# User inputs
user_vals = []

cols = st.columns(3)
for idx, f in enumerate(features):
    with cols[idx % 3]:
        value = st.number_input(f, min_value=0.0, max_value=100.0, value=5.0)
        user_vals.append(value)

if st.button("Predict Personality Cluster"):
    
    user_array = np.array(user_vals).reshape(1, -1)
    user_scaled = scaler.transform(user_array)

    # Hard cluster prediction
    hard_cluster = rfc.predict(user_scaled)[0]

    # Soft probabilistic cluster
    raw_soft = multi.predict(user_scaled)
    prob_soft = temperature_softmax(raw_soft, T=1.5)[0]
    soft_cluster = int(np.argmax(prob_soft))

    st.subheader("🔮 Prediction Results")
    st.write(f"**Hard Cluster (KMeans → RFC):** {hard_cluster}")
    st.write(f"**Soft Cluster (GMM → CatBoost):** {soft_cluster}")

    st.subheader("📊 Probability Breakdown")
    for i, p in enumerate(prob_soft):
        st.write(f"Cluster {i}:  **{p:.4f}**")
