from crewai import Agent, Task

def create_manager_agent(llm):
    return Agent(
        role='Supply Chain Manager',
        goal='Create a strategic action plan based on audits and investigations.',
        backstory="Senior supply chain executive balancing risk, cost, and relationships.",
        llm=llm, verbose=True
    )

def create_manager_task(agent, context_tasks):
    return Task(
        description="""
        1. EXECUTIVE SUMMARY: Key findings + urgency level (2-3 sentences).
        2. SUPPLIER RISK MATRIX:
           | Supplier | Internal Risk | External Risk | Action: RETAIN/MONITOR/REPLACE/URGENT |
        3. PRIORITIZED ACTION PLAN: Rank by urgency. For each: What, Why, Who.
        4. MONITORING RECOMMENDATIONS: KPIs to track for these suppliers.
        """,
        expected_output="Strategic report with prioritized action plan in Markdown",
        agent=agent,
        context=context_tasks   # receives output from Auditor + Investigator
    )
