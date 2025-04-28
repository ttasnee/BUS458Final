import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Data Science Salary Predictor", layout="wide")

# --- LOAD MODEL AND FEATURES ---
@st.cache_resource
def load_pipeline_and_features():
    model = joblib.load("best_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_pipeline_and_features()

# --- USER INPUTS ---
st.title("Data Science Salary Predictor")

st.sidebar.header("Input Features")

# Country
country = st.sidebar.selectbox('Country', [
    "United States of America", "India", "Germany", "United Kingdom", "Canada", 
    "Brazil", "France", "Japan", "Australia", "Spain", "Other"
])

# Coding Experience (ordinal)
coding_exp_options = [
    "I have never written code",
    "< 1 years",
    "1-3 years",
    "3-5 years",
    "5-10 years",
    "10-20 years",
    "20+ years"
]
coding_exp = st.sidebar.selectbox('Coding Experience', coding_exp_options)

# ML Experience (ordinal)
ml_exp_options = [
    "I do not use machine learning methods",
    "Under 1 year",
    "1-2 years",
    "2-3 years",
    "3-4 years",
    "4-5 years",
    "5-10 years",
    "10-20 years",
    "20+ years"
]
ml_exp = st.sidebar.selectbox('ML Experience', ml_exp_options)
# ml_exp_cat = st.sidebar.selectbox('ML Experience (categorical)', ml_exp_options)  # For Q16 dummy


# Q27 (cloud usage, ordinal)
cloud_usage_options = [
    "No cloud usage", "Basic usage", "Moderate usage", "Extensive usage", "Enterprise level"
]
cloud_usage = st.sidebar.selectbox('Cloud Platform Usage Level (Q27)', cloud_usage_options)


# Q30 (money spent, ordinal)
money_spent_options = [
    "$0 ($USD)", "$1-$99", "$100-$999", "$1000-$9,999", "$10,000-$99,999", "$100,000+"
]
money_spent = st.sidebar.selectbox('Money Spent on ML/Cloud (Q30)', money_spent_options)


# Age (ordinal)
age_options = ["18-21", "22-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-69", "70+"]
age = st.sidebar.selectbox('Age', age_options)

# Education (ordinal)
education_options = [
    "No formal education past high school",
    "Some college/university study without earning a bachelor's degree",
    "Bachelor's degree",
    "Master's degree",
    "Doctoral degree",
    "Professional degree",
    "Other"
]
education = st.sidebar.selectbox('Education Level', education_options)

# Q23 job title
job_title = st.sidebar.selectbox('Job Title', [
    "Data Scientist", "Software Engineer", "Data Analyst", "Research Scientist", 
    "Machine Learning Engineer", "Data Engineer", "Teacher / Professor", "Other"
])

# Categorical dummies (top 10 from each during training)
gender = st.sidebar.selectbox('Gender', ["Man", "Woman", "Prefer not to say", "Nonbinary", "Other"])


industry = st.sidebar.selectbox('Industry', [
    "Computers/Technology", "Academics/Education", "Finance", "Healthcare", 
    "Insurance/Risk Assessment", "Retail/Sales", "Other"
])
cloud_platform = st.sidebar.selectbox('Cloud Platform', [
    "Amazon Web Services (AWS)", "Microsoft Azure", "Google Cloud Platform (GCP)", "None", "Other"
])

# --- ORDINAL ENCODING ---
age_map = {v: i+1 for i, v in enumerate(age_options)}
education_map = {
    "No formal education past high school": 1,
    "Some college/university study without earning a bachelor's degree": 2,
    "Bachelor's degree": 3,
    "Master's degree": 4,
    "Doctoral degree": 5,
    "Professional degree": 4,
    "Other": 2
}
coding_exp_map = {v: i for i, v in enumerate(coding_exp_options)}
ml_exp_map = {v: i for i, v in enumerate(ml_exp_options)}
money_spent_map = {v: i for i, v in enumerate(money_spent_options)}
cloud_usage_map = {v: i for i, v in enumerate(cloud_usage_options)}

# --- BUILD USER DATA DICT ---
user_data = {
    'age_ordinal': age_map[age],
    'education_ordinal': education_map[education],
    'coding_exp_ordinal': coding_exp_map[coding_exp],
    'ml_exp_ordinal': ml_exp_map[ml_exp],
    'q30_ordinal': money_spent_map[money_spent],
    'q27_ordinal': cloud_usage_map[cloud_usage],
}

# Add dummies for top categories (set to 1 if selected, else 0)
dummy_map = {
    'Q3_Man': gender == "Man",
    'Q3_Woman': gender == "Woman",
    'Q4_United States of America': country == "United States of America",
    'Q4_India': country == "India",
    'Q4_Germany': country == "Germany",
    'Q4_United Kingdom': country == "United Kingdom",
    'Q23_Data Scientist': job_title == "Data Scientist",
    'Q23_Software Engineer': job_title == "Software Engineer",
    'Q23_Data Analyst': job_title == "Data Analyst",
    'Q24_Computers/Technology': industry == "Computers/Technology",
    'Q24_Academics/Education': industry == "Academics/Education",
    'Q24_Finance': industry == "Finance",
    'Q27_Amazon Web Services (AWS)': cloud_platform == "Amazon Web Services (AWS)",
    'Q27_Microsoft Azure': cloud_platform == "Microsoft Azure",
    'Q27_Google Cloud Platform (GCP)': cloud_platform == "Google Cloud Platform (GCP)"
}
for col in dummy_map:
    user_data[col] = int(dummy_map[col])

# --- ALIGN FEATURES FOR PREDICTION ---
# Create DataFrame with all feature columns used in training, fill with 0
input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
# Overwrite with user-provided values where available
for col, val in user_data.items():
    if col in input_df.columns:
        input_df.at[0, col] = val

# --- PREDICTION ---
if st.sidebar.button('Predict Salary'):
    log_prediction = model.predict(input_df)[0]
    prediction = np.expm1(log_prediction)
    st.header('Salary Prediction')
    st.write(f"The predicted annual salary is: **${prediction:,.2f}**")
    lower_bound = np.expm1(log_prediction - 0.5)
    upper_bound = np.expm1(log_prediction + 0.5)
    st.write(f"Estimated range: **${lower_bound:,.2f} - ${upper_bound:,.2f}**")
    st.success("Prediction complete!")

# --- ABOUT ---
st.markdown("---")
st.markdown("**Data Science Salary Predictor by Tasfia Tasneem | Based on Kaggle Survey 2022**")
