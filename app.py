import streamlit as st
import pandas as pd
import os
import sys
from crewai import Crew, LLM
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.analysis.eda import run_eda, display_eda_section
from src.analysis.feature_engineering import engineer_features
from src.analysis.risk_scoring import calculate_risk_metrics
from src.visualization.charts import create_risk_dashboard
from src.models.predictive import train_predictive_model
from src.agents.auditor import create_auditor_agent, create_auditor_task
from src.agents.investigator import create_investigator_agent, create_investigator_task
from src.agents.manager import create_manager_agent, create_manager_task

st.set_page_config(page_title="Agentic Supply Chain Risk Intelligence System", layout="wide")

st.sidebar.header(" Configuration")
serper_api_key = st.sidebar.text_input("Serper API Key", value=os.environ.get("SERPER_API_KEY", ""), type="password")
google_api_key = st.sidebar.text_input("Google API Key", value=os.environ.get("GOOGLE_API_KEY", ""), type="password")

os.environ["SERPER_API_KEY"] = serper_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key

st.title(" Agentic Supply Chain Risk Intelligence System")
st.markdown("""
This system uses **Statistical Analysis** and **Multi-Agent AI** to audit supply chain risks.
""")
uploaded_file = st.file_uploader("Upload Supplier Data (CSV)", type=["csv"])
if st.checkbox("Use Demo Data"):
    if os.path.exists("data/sample_suppliers.csv"):
        uploaded_file = "data/sample_suppliers.csv"
    elif os.path.exists("../data/sample_suppliers.csv"):
        uploaded_file = "../data/sample_suppliers.csv"
    else:
        st.warning("Demo data not found.")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        display_eda_section(df)
        with st.spinner("Running Advanced Risk Analysis..."):
            df_enriched = engineer_features(df)
            df_scored = calculate_risk_metrics(df_enriched)
            if 'caused_disruption' in df_scored.columns: 
                df_scored, importance = train_predictive_model(df_scored)

        st.subheader(" AI Risk Assessment Metrics")
        cols_to_show = [c for c in ['supplier_name', 'composite_risk_score', 'predicted_risk_probability', 'supplier_tier'] if c in df_scored.columns]
        st.dataframe(df_scored[cols_to_show].head())
        st.subheader(" Risk Visualizations")
        charts = create_risk_dashboard(df_scored)

        c1, c2 = st.columns(2)
        if 'histogram' in charts:
            c1.plotly_chart(charts['histogram'], use_container_width=True)
        if 'scatter' in charts:
            c2.plotly_chart(charts['scatter'], use_container_width=True)

        if 'pareto' in charts:
            st.plotly_chart(charts['pareto'], use_container_width=True)

        if 'predicted_risk_probability' in df_scored.columns:
            st.metric("Avg Predicted Risk Probability", f"{df_scored['predicted_risk_probability'].mean():.2f}")

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        st.stop()
    st.divider()
    st.header(" Agentic Investigation")

    if st.button(" Kickoff Agents"):
        if not google_api_key or not serper_api_key:
            st.error("Please provide API Keys in the sidebar.")
            st.stop()

        with st.spinner('The Crew is working... agents are auditing, investigating, and planning.'):
            try:
                llm = LLM(
                    model="gemini/gemini-2.5-flash",
                    api_key=google_api_key
                )
                auditor_agent = create_auditor_agent(llm, df=df_scored)
                investigator_agent = create_investigator_agent(llm)
                manager_agent = create_manager_agent(llm)
                task1 = create_auditor_task(auditor_agent)
                task2 = create_investigator_task(investigator_agent, [task1])
                task3 = create_manager_task(manager_agent, [task1, task2])
                crew = Crew(
                    agents=[auditor_agent, investigator_agent, manager_agent], 
                    tasks=[task1, task2, task3], 
                    verbose=True
                )
                result = crew.kickoff()

                st.success("Mission Complete!")
                st.subheader("Final Strategic Report")
                st.markdown(result)

            except Exception as e:
                import traceback
                st.error(f"An agent error occurred: {e}")
                st.code(traceback.format_exc())

else:
    st.info("Please upload a CSV file or select Demo Data to begin.")
