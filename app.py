#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import os
import pandas as pd
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool


st.set_page_config(page_title="Agentic Supply Chain Risk Intelligence System")

st.sidebar.header("⚙️ Configuration")
serper_api_key = st.sidebar.text_input("Serper API Key", value="PASTE_HERE", type="password")
google_api_key = st.sidebar.text_input("Google API Key", value="PASTE_HERE", type="password")


os.environ["SERPER_API_KEY"] = serper_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key

st.title("Agentic Supply Chain Risk Intelligence System")
st.markdown("Upload your supplier CSV, and the **AI Crew** will audit delays, check external news risks, and propose an action plan.")


uploaded_file = st.file_uploader("Upload Supplier Data (CSV)", type=["csv"])
TEMP_CSV_PATH = "internal_suppliers.csv"

if uploaded_file is not None:
   
    df = pd.read_csv(uploaded_file)
    df.to_csv(TEMP_CSV_PATH, index=False)
    
    st.subheader(" Data Preview")
    st.dataframe(df.head())
else:
    st.info("Please upload a CSV file to begin.")
    st.stop() 

if google_api_key:
    gemini_llm = LLM(
        model="gemini/gemini-2.5-flash",
        api_key=google_api_key
    )
else:
    st.error("Please enter a Google API Key in the sidebar.")
    st.stop()

class SupplierRiskAnalyzer(BaseTool):
    name: str = "Analyze Supplier Risk CSV"
    description: str = "Reads 'internal_suppliers.csv' to find suppliers with high delays or risk scores."

    def _run(self, query: str) -> str:

        filename = TEMP_CSV_PATH
        try:
            if not os.path.exists(filename):
                return "Error: File not found."

            df = pd.read_csv(filename)
            required_cols = ['avg_delay_days', 'risk_score']
            

            if not all(col in df.columns for col in required_cols):
                return f"Error: Missing columns. Expected {required_cols}, found {list(df.columns)}"

 
            risky = df[(df['avg_delay_days'] > 5) | (df['risk_score'] > 50)]

            if risky.empty:
                return "Analysis Complete: No high-risk suppliers found."

            return f"Found {len(risky)} risky suppliers:\n{risky.to_string()}"
            
        except Exception as e:
            return f"Error processing CSV: {str(e)}"


csv_tool = SupplierRiskAnalyzer()
search_tool = SerperDevTool()


auditor = Agent(
    role='Supply Chain Auditor',
    goal='Identify late suppliers from the internal data.',
    backstory="You strictly rely on the CSV file to find bottlenecks.",
    tools=[csv_tool],
    llm=gemini_llm,
    verbose=True
)

investigator = Agent(
    role='Risk Investigator',
    goal='Find external reasons for the delays.',
    backstory="When the Auditor finds a delay in a city, you check Google News for that location.",
    tools=[search_tool],
    llm=gemini_llm,
    verbose=True
)

manager = Agent(
    role='Supply Chain Manager',
    goal='Write an action plan.',
    backstory="You recommend whether to fire or keep the suppliers.",
    llm=gemini_llm,
    verbose=True
)


task1 = Task(
    description="Analyze 'internal.csv'. List any supplier with Avg_Delay_Days > 5.", 
    expected_output="List of suppliers", 
    agent=auditor
)

task2 = Task(
    description="Search news for the locations of the late suppliers.", 
    expected_output="News summary", 
    agent=investigator, 
    context=[task1]
)

task3 = Task(
    description="Write a strategic recommendation report.", 
    expected_output="Final Report in Markdown", 
    agent=manager, 
    context=[task1, task2]
)


if st.button("🚀 Run AI Analysis"):
    with st.spinner('The Crew is working... (This may take ~30 seconds)'):
        try:
            crew = Crew(
                agents=[auditor, investigator, manager], 
                tasks=[task1, task2, task3], 
                verbose=True
            )
            result = crew.kickoff()
            
            st.success("Analysis Complete!")
            st.subheader("Final Strategic Report")
            st.markdown(result)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")


# In[ ]:




