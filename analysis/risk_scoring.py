import pandas as pd
from scipy import stats

def calculate_risk_metrics(df):
    """Adds statistical risk flags to the dataframe."""
    df = df.copy()

    if 'avg_delay_days' not in df.columns or 'risk_score' not in df.columns:
        return df

    
    df['is_delay_outlier_percentile'] = df['avg_delay_days'] > df['avg_delay_days'].quantile(0.9)
    df['is_risk_outlier_percentile']  = df['risk_score']     > df['risk_score'].quantile(0.9)

    
    df['delay_zscore'] = stats.zscore(df['avg_delay_days']) if df['avg_delay_days'].std() > 0 else 0
    df['risk_zscore']  = stats.zscore(df['risk_score'])     if df['risk_score'].std() > 0     else 0

    df['is_delay_anomaly_zscore'] = df['delay_zscore'] > 2
    df['is_risk_anomaly_zscore']  = df['risk_zscore']  > 2


    df['is_statistically_risky'] = (
        df['is_delay_outlier_percentile'] |
        df['is_risk_outlier_percentile']  |
        df['is_delay_anomaly_zscore']     |
        df['is_risk_anomaly_zscore']
    )

    return df
