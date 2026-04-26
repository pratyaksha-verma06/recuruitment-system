import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="AI Recruitment Dashboard")

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
    job_model = pickle.load(open("job_model.pkl", "rb"))
    salary_model = pickle.load(open("salary_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    kmeans = pickle.load(open("kmeans.pkl", "rb"))
    salary_features = pickle.load(open("features.pkl", "rb")) 
    

    df = pd.read_csv("resume_dataset_with_salary.csv")
    
    if 'cluster' not in df.columns:
        skills_series = df['user_skill'].str.split(';').apply(lambda x: [i.strip().lower() for i in x])
        encoded_skills = mlb.transform(skills_series)
        df['cluster'] = kmeans.predict(encoded_skills)
    
    return mlb, job_model, salary_model, scaler, kmeans, salary_features, df

try:
    mlb, job_model, salary_model, scaler, kmeans, salary_features, df = load_assets()
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
    st.title("🔍 New Candidate Selection")
    st.markdown("Enter candidate details directly to evaluate fit based on historical benchmarks.")
    
    with st.form("screening_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            edu = st.selectbox("Highest Education Level", ["Bachelor", "Master", "PhD"])
            job_description = st.text_area("Candidate Skills (comma separated)", "Python, SQL, Machine Learning")
            exp = st.number_input("Years of Professional Experience", 0, 25, 5)

        with col2:
            score = st.slider("Keyword Match Score (%)", 0, 100, 70)
            roles_list = sorted(df['job_title'].unique().tolist())
            applied_role = st.selectbox("Apply for Role", roles_list)
            salary_range = st.selectbox("Candidate Salary Expectation", ["70k-80k", "80k-90k", "90k-100k", "100k+"])

        if st.form_submit_button("Analyze Candidate Fit", use_container_width=True):
            st.divider()
            
            is_selected = score >= 70
            
            if is_selected:
                st.success(f"### Result: SELECTED")
                st.write(f"The candidate is a strong match for the **{applied_role}** position.")
            else:
                st.error("### Result: NOT RECOMMENDED")
                st.write("The candidate's score is below the selection threshold.")


elif choice == "Salary Estimator":
    st.title("💰 Salary Benchmark Tool")
    st.markdown("Predict market-competitive salaries based on candidate profile.")
    
    col1, col2 = st.columns(2)
    with col1:
        exp_input = st.slider("Years of Professional Experience", 0, 25, 5)
    with col2:
        edu_input = st.selectbox("Highest Education Level", ["Bachelor", "Master", "PhD"])
    
    if st.button("Generate Estimate", use_container_width=True):
        edu_map = {"Bachelor": 1, "Master": 2, "PhD": 3}
        input_data = pd.DataFrame([[exp_input, edu_map[edu_input]]], columns=salary_features)
        
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