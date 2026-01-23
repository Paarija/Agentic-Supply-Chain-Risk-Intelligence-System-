from crewai import Agent, Task

def create_manager_agent(llm):
    agent = Agent(
        role='Supply Chain Manager',
        goal='Formulate a strategic action plan based on internal audits and external investigations.',
        backstory="You are a senior supply chain executive. You balance risk, cost, and relationships to make tough decisions.",
        llm=llm,
        verbose=True
    )
    return agent

def create_manager_task(agent, context_tasks):
    return Task(
        description="""
        Create a comprehensive strategic recommendation report:
        
        1. EXECUTIVE SUMMARY (2-3 sentences)
           - Key findings
           - Urgency level
        
        2. SUPPLIER RISK MATRIX:
           Categorize each risky supplier:
           | Supplier | Internal Risk | External Risk | Action Consideration |
           
           Action options: RETAIN / MONITOR / REPLACE / URGENT_ACTION
        
        3. PRIORITIZED ACTION PLAN:
           Rank actions by urgency.
           For each action:
           - What: Specific action
           - Why: Justification (cite the data or news)
           - Who: Responsible party
        
        4. MONITORING RECOMMENDATIONS:
           - What specific KPIs should we watch closely for these suppliers?
        """,
        expected_output="Strategic report with prioritized action plan in Markdown",
        agent=agent,
        context=context_tasks
    )
