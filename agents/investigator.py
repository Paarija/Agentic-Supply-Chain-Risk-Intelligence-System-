from crewai import Agent, Task
from crewai_tools import SerperDevTool

def create_investigator_agent(llm):
    return Agent(
        role='Risk Investigator',
        goal='Investigate external factors (strikes, weather, politics) for flagged locations.',
        backstory="OSINT expert who verifies if delays are caused by external disruptions.",
        tools=[SerperDevTool()],   # Google Search via Serper API
        llm=llm, verbose=True
    )

def create_investigator_task(agent, context_tasks):
    return Task(
        description="""
        For each risky supplier from the Auditor:
        1. Search: "[Location] supply chain OR disruption OR strike OR weather"
        2. Return a table: | Supplier | Location | External Factor | Source | Confidence |
        3. If no news found, provide a baseline geographic risk assessment.
        """,
        expected_output="Structured external risk assessment per supplier",
        agent=agent,
        context=context_tasks   # receives output from the Auditor task
    )
