import streamlit as st
import pandas as pd
import pickle
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Salary Impact Predictor",
    page_icon="🚀",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #eef2f3, #d9e4f5);
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(90deg, #007BFF, #00C6FF);
    animation: fadeIn 2s ease-in-out;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.15);
    transition: transform 0.3s ease;
}
.card:hover {
    transform: scale(1.02);
}

.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
    background: linear-gradient(90deg, #007BFF, #00C6FF);
    color: white;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 10px rgba(0,123,255,0.4);
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
import os

model_path = "model.pkl"

# If model doesn't exist, train it
if not os.path.exists(model_path):
    st.warning("⚠️ Training model... this may take a moment on first load.")
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    
    df = pd.read_csv('ai_job_impact.csv')
    features = ['Age', 'Years_Experience', 'Salary_Before_AI', 'Work_Hours_Per_Week', 'Job_Satisfaction', 'Education_Level', 'Industry']
    target = 'Salary_After_AI'
    
    X = df[features]
    y = df[target]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), ['Age', 'Years_Experience', 'Salary_Before_AI', 'Work_Hours_Per_Week', 'Job_Satisfaction']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Education_Level', 'Industry'])
        ])
    
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    model.fit(X, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    st.success("✅ Model trained and saved!")
else:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🚀 AI Salary Impact Predictor</div>', unsafe_allow_html=True)
st.markdown("### Predict how AI may impact your future salary using your professional profile.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")
show_chart = st.sidebar.checkbox("Show Salary Comparison Chart", True)
show_confidence = st.sidebar.checkbox("Show Confidence Score", True)

# ---------------- MAIN FORM ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    years_exp = st.slider("Years of Experience", 0, 40, 5)
    salary_before = st.number_input("Salary Before AI", 20000, 200000, 50000)

with col2:
    hours = st.slider("Work Hours Per Week", 10, 80, 40)
    satisfaction = st.slider("Job Satisfaction", 1, 10, 5)
    education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD", "High School"])
    industry = st.selectbox("Industry", [
        "IT", "Healthcare", "Finance",
        "Manufacturing", "Education", "Marketing"
    ])

# ---------------- BUTTONS ----------------
col_btn1, col_btn2 = st.columns(2)

predict = col_btn1.button("🔮 Predict Salary")
reset = col_btn2.button("🔄 Reset")

if reset:
    st.rerun()

# ---------------- PREDICTION ----------------
if predict:
    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.02)
        progress.progress(i + 1)

    input_data = pd.DataFrame({
        'Age': [age],
        'Years_Experience': [years_exp],
        'Salary_Before_AI': [salary_before],
        'Work_Hours_Per_Week': [hours],
        'Job_Satisfaction': [satisfaction],
        'Education_Level': [education],
        'Industry': [industry]
    })

    prediction = model.predict(input_data).item()
    change = prediction - salary_before

    st.balloons()

    # ----------- METRICS -----------
    st.markdown("## 📊 Prediction Results")
    m1, m2, m3 = st.columns(3)

    m1.metric("Current Salary", f"${salary_before:,.0f}")
    m2.metric("Predicted Salary", f"${prediction:,.0f}", f"{change:,.0f}")
    m3.metric("Growth %", f"{(change/salary_before)*100:.2f}%")

    # ----------- CONFIDENCE BAR -----------
    if show_confidence:
        confidence = min(max((years_exp * 2 + satisfaction * 5), 50), 95)
        st.markdown("### Confidence Score")
        st.progress(confidence)
        st.write(f"Model Confidence: **{confidence}%**")

    # ----------- CHART -----------
    if show_chart:
        chart_data = pd.DataFrame({
            'Salary': [salary_before, prediction]
        }, index=['Before AI', 'Predicted'])

        st.markdown("### Salary Comparison")
        st.bar_chart(chart_data)

    # ----------- FINAL MESSAGE -----------
    if prediction > salary_before:
        st.success("🎉 Great news! AI is predicted to increase your salary.")
    else:
        st.warning("⚠️ AI may negatively affect salary growth in this profile.")