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
    max_delay = df['avg_delay_days'].max()
    min_delay = df['avg_delay_days'].min()
    ptp_delay = max_delay - min_delay
    if ptp_delay == 0: ptp_delay = 1

    max_risk = df['risk_score'].max()
    min_risk = df['risk_score'].min()
    ptp_risk = max_risk - min_risk
    if ptp_risk == 0: ptp_risk = 1

    norm_delay = (df['avg_delay_days'] - min_delay) / ptp_delay
    norm_risk = (df['risk_score'] - min_risk) / ptp_risk
    df['composite_risk_score'] = (0.6 * norm_delay) + (0.4 * norm_risk)

    mask_statistical = (
        (df['is_delay_outlier_percentile']) | 
        (df['is_risk_outlier_percentile']) |
        (df['is_delay_anomaly_zscore']) | 
        (df['is_risk_anomaly_zscore'])
    )

    df['is_statistically_risky'] = mask_statistical

    return df
