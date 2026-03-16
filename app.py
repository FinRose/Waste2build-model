import streamlit as st
import joblib
import pandas as pd

# 1. PAGE SETUP
st.set_page_config(page_title="Waste2Build AI", page_icon="♻️")
st.title("♻️ Waste2Build: Onboarding Intelligence")

# --- ADD THE NEW LINE HERE ---
st.info("Goal: Supporting SDG 15 (Life on Land) by identifying reliable waste management partners.")
# -----------------------------

st.write("Predict if a new waste seller is a 'High Potential' partner.")

# 2. LOAD THE SAVED BRAIN (The .pkl file you made with joblib)
@st.cache_resource # This makes the app fast
def load_my_model():
    return joblib.load('waste2build_model.pkl')

model = load_my_model()

# 3. USER INTERFACE (The Sidebar)
st.sidebar.header("New Seller Characteristics")
freq = st.sidebar.slider("Drops-offs per Month", 1, 30, 5)
accuracy = st.sidebar.slider("Reporting Accuracy (0 to 1)", 0.0, 1.0, 0.8)
total_kg = st.sidebar.number_input("Last Month Volume (kg)", 0.0, 500.0, 10.0)
purity = st.sidebar.slider("Material Purity (0 to 1)", 0.0, 1.0, 0.7)

# 4. PREDICTION LOGIC
if st.button("Analyze Seller Potential"):
    # Arrange data exactly like our training dataframe
    input_df = pd.DataFrame([{
        'frequency_per_month': freq,
        'avg_accuracy_score': accuracy,
        'total_kg_last_month': total_kg,
        'material_purity_rate': purity
    }])
    
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    # 5. DISPLAY RESULTS
    st.subheader("Results")
    if prediction[0] == 1:
        st.success(f"🌟 High Potential Seller! (Match Score: {probability:.1%})")
        st.balloons()
    else:
        st.warning(f"⚖️ Standard Seller (Match Score: {probability:.1%})")