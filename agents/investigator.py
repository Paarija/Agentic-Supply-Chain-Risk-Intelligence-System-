from crewai import Agent, Task
from crewai_tools import SerperDevTool

def create_investigator_agent(llm):
    search_tool = SerperDevTool()
    
    agent = Agent(
        role='Risk Investigator',
        goal='Investigate external factors (strikes, weather, politics) for flagged locations.',
        backstory="You are an expert in OSINT (Open Source Intelligence). You verify if internal delays are caused by external disruptions.",
        tools=[search_tool],
        llm=llm,
        verbose=True
    )
    return agent

def create_investigator_task(agent, context_tasks):
    return Task(
        description="""
        For each risky supplier identified by the Auditor, investigate external factors:
        
        1. SEARCH PARAMETERS:
           - Query: "[Location] supply chain OR logistics OR disruption OR strike OR weather"
           - Time range: Last 30 days implied (unless tool supports specific range)
        
        2. STRUCTURED OUTPUT FOR EACH SUPPLIER:
           | Supplier | Location | External Factor | Source | Confidence |
           
           Factors to investigate:
           - Labor strikes/disputes
           - Weather events (floods, storms)
           - Political instability
           - Port congestion
        
        3. FALLBACK REASONING:
           If no news found, provide baseline geographic risk assessment based on the location.
        """,
        expected_output="Structured external risk assessment per supplier",
        agent=agent,
        context=context_tasks
    )
