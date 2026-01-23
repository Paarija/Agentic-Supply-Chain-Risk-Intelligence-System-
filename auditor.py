from crewai import Agent, Task
from crewai.tools import BaseTool
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.analysis.risk_scoring import calculate_risk_metrics
from src.analysis.feature_engineering import engineer_features

class SupplierRiskAnalyzer(BaseTool):
    name: str = "Analyze Supplier Risk CSV"
    description: str = "Reads internal suppliers data and finds statistical risks."
    data_frame: pd.DataFrame = None
    file_path: str = "internal_suppliers.csv"

    def __init__(self, data_frame=None):
        super().__init__()
        self.data_frame = data_frame

    def _run(self, query: str = None) -> str:
        try:
            if self.data_frame is not None:
                df = self.data_frame
            elif os.path.exists(self.file_path):
                df = pd.read_csv(self.file_path)
            else:
                return "Error: Data source not found (neither in-memory DataFrame nor CSV)."
            if 'risk_score' not in df.columns:
                 df = engineer_features(df)
                 df = calculate_risk_metrics(df)
            if 'is_statistically_risky' in df.columns:
                risky = df[df['is_statistically_risky']]
            else:
                risky = df[(df['avg_delay_days'] > 5) | (df['risk_score'] > 50)]

            if risky.empty:
                return "Analysis Complete: No statistically significant high-risk suppliers found."

            cols = ['supplier_name', 'location', 'avg_delay_days', 'risk_score', 'composite_risk_score', 'is_delay_outlier_percentile', 'is_risk_anomaly_zscore']
            available_cols = [c for c in cols if c in risky.columns]

            return f"Found {len(risky)} risky suppliers:\n{risky[available_cols].to_string()}"

        except Exception as e:
            return f"Error processing data: {str(e)}"

def create_auditor_agent(llm, df=None):
    csv_tool = SupplierRiskAnalyzer(data_frame=df)

    agent = Agent(
        role='Supply Chain Auditor',
        goal='Identify suppliers that pose specific statistical risks using the enhanced analysis tool.',
        backstory="You are a data-driven auditor who looks beyond simple thresholds. You use statistical outliers and composite scores to flag genuine risks.",
        tools=[csv_tool],
        llm=llm,
        verbose=True
    )
    return agent

def create_auditor_task(agent):
    return Task(
        description="""
        Analyze 'internal_suppliers.csv' with the following steps:

        1. STATISTICAL SUMMARY:
           - Use the Analysis Tool to process the data.
           - The tool will automatically calculate composite scores and statistical outliers.

        2. RISK IDENTIFICATION:
           - Identify which suppliers are flagged as risky.
           - Pay attention to 'composite_risk_score' and z-score flags.

        3. OUTPUT FORMAT:
           Return a structured table with columns:
           | Rank | Supplier | Location | Delay (days) | Risk Score | Composite Risk | Statistical Flags |

        4. STATISTICAL JUSTIFICATION:
           For each flagged supplier, explain WHY they are risky using the statistical flags provided 
           (e.g., "Flagged as Delay Outlier (90th percentile)" or "High Z-Score").
        """,
        expected_output="Ranked risk table with statistical justification",
        agent=agent
    )
