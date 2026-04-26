import pandas as pd
import numpy as np
from scipy import stats

def calculate_risk_metrics(df):
    """
    Applies statistical methods to identify risky suppliers.
    Adds scoring columns to the dataframe.
    """
    df = df.copy()
    if 'avg_delay_days' not in df.columns or 'risk_score' not in df.columns:
        return df
    delay_p90 = df['avg_delay_days'].quantile(0.9)
    risk_p90 = df['risk_score'].quantile(0.9)

    df['is_delay_outlier_percentile'] = df['avg_delay_days'] > delay_p90
    df['is_risk_outlier_percentile'] = df['risk_score'] > risk_p90
    if df['avg_delay_days'].std() > 0:
        df['delay_zscore'] = stats.zscore(df['avg_delay_days'])
    else:
        df['delay_zscore'] = 0

    if df['risk_score'].std() > 0:
        df['risk_zscore'] = stats.zscore(df['risk_score'])
    else:
        df['risk_zscore'] = 0
    df['is_delay_anomaly_zscore'] = df['delay_zscore'] > 2
    df['is_risk_anomaly_zscore'] = df['risk_zscore'] > 2
    

    mask_statistical = (
        (df['is_delay_outlier_percentile']) | 
        (df['is_risk_outlier_percentile']) |
        (df['is_delay_anomaly_zscore']) | 
        (df['is_risk_anomaly_zscore'])
    )

    df['is_statistically_risky'] = mask_statistical

    return df
