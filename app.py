import streamlit as st
import pandas as pd
import os
from crewai import Crew, LLM


from analysis.eda import run_eda, display_eda_section
from analysis.feature_engineering import engineer_features
from analysis.risk_scoring import calculate_risk_metrics
from charts import create_risk_dashboard
from predictive import train_predictive_model
from agents.auditor import create_auditor_agent, create_auditor_task
from agents.investigator import create_investigator_agent, create_investigator_task
from agents.manager import create_manager_agent, create_manager_task


st.set_page_config(page_title="Supply Chain Risk System", layout="wide")


st.sidebar.header("Configuration")
serper_key = st.sidebar.text_input("Serper API Key", type="password")
google_key = st.sidebar.text_input("Google API Key", type="password")
os.environ["SERPER_API_KEY"] = serper_key
os.environ["GOOGLE_API_KEY"] = google_key


st.title("Agentic Supply Chain Risk Intelligence System")

uploaded_file = st.file_uploader("Upload Supplier Data (CSV)", type=["csv"])
if st.checkbox("Use Demo Data"):
    uploaded_file = "data/sample_suppliers.csv"


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    display_eda_section(df)

    
    df = engineer_features(df)          
    df = calculate_risk_metrics(df)     

 
    importance = {}
    if 'caused_disruption' in df.columns:
        df, importance = train_predictive_model(df)

    
    st.subheader("Risk Assessment Metrics")
    show_cols = [c for c in ['supplier_name', 'composite_risk_score',
                              'predicted_risk_probability', 'supplier_tier'] if c in df.columns]
    st.dataframe(df[show_cols].head())

    
    charts = create_risk_dashboard(df)
    c1, c2 = st.columns(2)
    if 'histogram' in charts: c1.plotly_chart(charts['histogram'])
    if 'scatter' in charts:   c2.plotly_chart(charts['scatter'])
    if 'pareto' in charts:    st.plotly_chart(charts['pareto'])

    
    if 'predicted_risk_probability' in df.columns:
        st.metric("Avg Predicted Risk", f"{df['predicted_risk_probability'].mean():.2f}")

   
    st.divider()
    st.header("Agentic Investigation")

    if st.button("Kickoff Agents"):
        if not google_key or not serper_key:
            st.error("Please provide API Keys in the sidebar.")
            st.stop()

        with st.spinner('Agents are working...'):
            
            llm = LLM(model="gemini/gemini-2.5-flash", api_key=google_key)

            
            auditor      = create_auditor_agent(llm, df=df)
            investigator = create_investigator_agent(llm)
            manager      = create_manager_agent(llm)

            
            task1 = create_auditor_task(auditor)
            task2 = create_investigator_task(investigator, [task1])
            task3 = create_manager_task(manager, [task1, task2])

            
            crew = Crew(agents=[auditor, investigator, manager],
                        tasks=[task1, task2, task3], verbose=True)
            result = crew.kickoff()

            st.success("Mission Complete!")
            st.markdown(result)
