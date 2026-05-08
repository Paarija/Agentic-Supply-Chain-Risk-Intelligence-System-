from crewai import Agent, Task
from crewai.tools import BaseTool
import pandas as pd
from analysis.risk_scoring import calculate_risk_metrics
from analysis.feature_engineering import engineer_features


class SupplierRiskAnalyzer(BaseTool):
    """Custom tool that lets the Auditor agent query the supplier dataframe."""
    name: str = "Analyze Supplier Risk CSV"
    description: str = "Reads supplier data and finds statistical risks."
    data_frame: pd.DataFrame = None

    def __init__(self, data_frame=None):
        super().__init__()
        self.data_frame = data_frame

    def _run(self, query: str = None) -> str:
        df = self.data_frame
        if df is None:
            return "Error: No data loaded."

        # Ensure risk columns exist
        if 'risk_score' not in df.columns:
            df = calculate_risk_metrics(engineer_features(df))

        # Filter to only risky suppliers
        if 'is_statistically_risky' in df.columns:
            risky = df[df['is_statistically_risky']]
        else:
            risky = df[(df['avg_delay_days'] > 5) | (df['risk_score'] > 50)]

        if risky.empty:
            return "No high-risk suppliers found."

        # Return a readable table of the risky ones
        cols = ['supplier_name', 'location', 'avg_delay_days', 'risk_score',
                'composite_risk_score', 'is_delay_outlier_percentile', 'is_risk_anomaly_zscore']
        show = [c for c in cols if c in risky.columns]
        return f"Found {len(risky)} risky suppliers:\n{risky[show].to_string()}"


def create_auditor_agent(llm, df=None):
    return Agent(
        role='Supply Chain Auditor',
        goal='Identify suppliers with statistical risks.',
        backstory="Data-driven auditor using statistical outliers and composite scores.",
        tools=[SupplierRiskAnalyzer(data_frame=df)],
        llm=llm, verbose=True
    )

def create_auditor_task(agent):
    return Task(
        description="""
        1. Use the Analysis Tool to process supplier data.
        2. Identify risky suppliers based on composite_risk_score and z-score flags.
        3. Return a ranked table: | Rank | Supplier | Location | Delay | Risk Score | Flags |
        4. For each, explain WHY they are risky (e.g., "90th percentile delay outlier").
        """,
        expected_output="Ranked risk table with statistical justification",
        agent=agent
    )
