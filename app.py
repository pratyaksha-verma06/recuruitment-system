import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="AI Recruitment Dashboard", layout="wide", page_icon="🤖")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stMetric p { color: black; }      
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource    
def load_assets():
    """Loads all models and datasets once to optimize performance."""

    mlb = pickle.load(open("mlb.pkl", "rb"))
    selection_model = pickle.load(open("model.pkl", "rb")) 
    selection_features = pickle.load(open("features.pkl", "rb")) 
    
    job_model = pickle.load(open("job_model.pkl", "rb"))
    salary_model = pickle.load(open("salary_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    kmeans = pickle.load(open("kmeans.pkl", "rb"))
    
    df = pd.read_csv("resume_dataset_with_salary.csv")
    
    if 'cluster' not in df.columns:
        skills_series = df['user_skill'].str.split(';').apply(lambda x: [i.strip().lower() for i in x])
        encoded_skills = mlb.transform(skills_series)
        df['cluster'] = kmeans.predict(encoded_skills)
    
    return mlb, selection_model, selection_features, job_model, salary_model, scaler, kmeans, df

try:
    mlb, selection_model, selection_features, job_model, salary_model, scaler, kmeans, df = load_assets()
except FileNotFoundError as e:
    st.error(f"Missing critical file: {e}. Ensure all .pkl and .csv files are in the app folder.")
    st.stop()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/912/912214.png", width=100)
    st.title("Recruitment AI")
    st.markdown("---")
    choice = st.radio("MAIN MENU", 
        ["Dashboard Home", "Resume Screening", "Salary Estimator", "Job Recommendation", "Role Clustering"])
    st.markdown("---")
    st.info("System Status: Online")

if choice == "Dashboard Home":
    st.title("📊 Talent Acquisition Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Resumes", len(df))
    col2.metric("Job Categories", df['job_title'].nunique())
    col3.metric("Avg Salary", f"${df['salary'].mean():,.0f}")
    
    st.subheader("Recent Applications")
    st.dataframe(df.head(10), use_container_width=True)

elif choice == "Resume Screening":
    st.title("🔍 Candidate Selection Portal")
    st.markdown("Evaluate candidate fit against specific job requirements using the selection model.")
    
    with st.form("screening_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            candidate_skills_raw = st.text_area("Candidate Skills (comma separated)", placeholder="e.g. Python, SQL, NLP")
            edu = st.selectbox("Highest Education Level", ["Bachelor", "Master", "PhD"])
            exp = st.number_input("Years of Professional Experience", 0, 25, 5)

        with col2:
            roles_list = sorted(df['job_title'].unique().tolist())
            applied_role = st.selectbox("Required Job", roles_list)

            role_reqs_raw = df[df['job_title'] == applied_role]['skills_required'].iloc[0]

        if st.form_submit_button("Analyze Candidate Fit", use_container_width=True):
            user_skills = [i.strip().lower() for i in candidate_skills_raw.split(",") if i.strip()]
            required_skills = [i.strip().lower() for i in role_reqs_raw.split(";")]
            
            user_set = set(user_skills)
            req_set = set(required_skills)
            matches = user_set.intersection(req_set)
            
            user_encoded = pd.DataFrame(mlb.transform([user_skills]), columns=mlb.classes_)
            req_encoded = pd.DataFrame(mlb.transform([required_skills]), columns=mlb.classes_)
            
            skill_gap = user_encoded - req_encoded
            matched_count = (user_encoded & req_encoded).sum(axis=1)
            
            edu_map = {"Bachelor": 1, "Master": 2, "PhD": 3}
            edu_val = edu_map.get(edu, 0)
            
            input_df = pd.concat([skill_gap, pd.DataFrame({
                "matched_skills": matched_count,
                "education": [edu_val],
                "experience": [exp]
            })], axis=1)
            
            input_df = input_df.reindex(columns=selection_features, fill_value=0)
            prediction = selection_model.predict(input_df)[0]
            probability = selection_model.predict_proba(input_df)[0][1]
            
            st.divider()
            
            if matches and prediction == 1:
                st.success(f"### RESUME SELECTED")
                st.write(f"The candidate is a strong match for **{applied_role}**.")
                st.write(f"**Matched Skills Found:** {', '.join(matches)}")
                st.metric("AI Confidence Score", f"{round(probability * 100, 2)}%")
            else:
                st.error(f"### RESUME NOT SELECTED")
                st.write(f"The candidate does not meet the requirements for **{applied_role}**.")
                if not matches:
                    st.warning("No overlapping skills found with the job requirements.")

elif choice == "Salary Estimator":
    st.title("💰 Salary Benchmark Tool")
    st.markdown("Predict market-competitive salaries based on experience and education.")
    
    col1, col2 = st.columns(2)
    with col1:
        exp_input = st.slider("Years of Professional Experience", 0, 25, 5)
    with col2:
        edu_input = st.selectbox("Highest Education Level", ["Bachelor", "Master", "PhD"])
    
    if st.button("Generate Estimate", use_container_width=True):
        edu_map = {"Bachelor": 1, "Master": 2, "PhD": 3}

        input_data = pd.DataFrame([[exp_input, edu_map[edu_input]]], columns=['experience', 'education'])
        
        input_scaled = scaler.transform(input_data)
        prediction = salary_model.predict(input_scaled)
        
        st.success(f"### Predicted Market Salary: ${prediction[0]:,.0f}")
        st.caption("Estimation based on Linear Regression analysis.")


elif choice == "Job Recommendation":
    st.title("🎯 AI Career Path Suggester")
    user_input = st.text_input("Enter Candidate Skills:", placeholder="e.g. Java, Spring Boot, SQL")
    
    if st.button("Predict Optimal Role", use_container_width=True):
        if user_input:
            user_skills = [i.strip().lower() for i in user_input.split(",")]
            user_encoded = mlb.transform([user_skills])
            

            prediction = job_model.predict(user_encoded)

            cluster_id = kmeans.predict(user_encoded)[0]
            
            st.subheader(f"Recommended Role: :blue[{prediction[0]}]")
            
            similar_roles = df[df['cluster'] == cluster_id]['job_title'].unique()
            st.write(f"**Other similar roles in this skill cluster:** {', '.join(similar_roles)}")
        else:
            st.warning("Please enter at least one skill.")

elif choice == "Role Clustering":
    st.title("🤝 Market Role Groupings")
    st.markdown("This view shows how job titles are grouped by the AI based on skill similarities.")
    
    cluster_view = df.groupby('cluster')['job_title'].unique().reset_index()
    st.dataframe(cluster_view, use_container_width=True)
