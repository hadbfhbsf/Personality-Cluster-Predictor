import streamlit as st
import numpy as np
import joblib

# ------------------------------
# LOAD MODELS
# ------------------------------
scaler = joblib.load("scaler.pkl")
multi = joblib.load("catboost_softlabel.pkl")
rfc = joblib.load("rf_classifier.pkl")

# ------------------------------
# FEATURES
# ------------------------------
features = [
    'Traditional Family', 'Extraversion', 'Neuroticism', 'Agreeableness', 
    'Concientiuosness', 'Openness', 'Factor A','Factor B','Factor C','Factor E',
    'Factor F','Factor G','Factor H','Factor I','Factor L','Factor M','Factor N',
    'Factor O','Factor Q1','Factor Q2','Factor Q3','Factor Q4',
    'E','I','S','N','F','T','J','P'
]

# ------------------------------
# SOFTMAX TEMPERATURE FUNCTION
# ------------------------------
def temperature_softmax(logits, T=1.5):
    logits = np.clip(logits, -20, 20)
    exp_vals = np.exp(logits / T)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("🧠 Personality Cluster Predictor")
st.write("Enter your **30 psychological trait scores** to get cluster predictions.")

# Collect user input
user_values = []

cols = st.columns(3)  # 3-column layout for neat input boxes
for i, f in enumerate(features):
    with cols[i % 3]:
        val = st.number_input(f, min_value=0.0, max_value=100.0, value=5.0)
        user_values.append(val)

user_array = np.array(user_values).reshape(1, -1)

# ------------------------------
# PREDICT BUTTON
# ------------------------------
if st.button("🔮 Predict My Personality Cluster"):
    
    # Scale input
    scaled = scaler.transform(user_array)

    # Hard label (RFC)
    hard_cluster = rfc.predict(scaled)[0]

    # Soft label (CatBoost)
    soft_raw = multi.predict(scaled)
    prob = temperature_softmax(soft_raw, T=1.5)[0]
    soft_cluster = np.argmax(prob)

    # Output
    st.subheader("🎯 RESULTS")

    st.write(f"**💡 Hard Cluster (KMeans → RFC):** `{hard_cluster}`")
    st.write(f"**🌈 Soft Cluster (GMM → CatBoost):** `{soft_cluster}`")

    st.subheader("📊 Cluster Probability Distribution")
    for i, p in enumerate(prob):
        st.write(f"- Cluster {i}: **{p:.4f}**")
