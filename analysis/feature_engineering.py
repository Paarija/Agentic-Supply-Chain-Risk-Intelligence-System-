import pandas as pd
import numpy as np

def categorize_region(location):
    if not isinstance(location, str):
        return "Unknown"

    location = location.lower()
    if any(x in location for x in ['new york', 'los angeles', 'chicago', 'usa', 'america']):
        return "North America"
    elif any(x in location for x in ['london', 'berlin', 'paris', 'europe', 'uk', 'germany']):
        return "Europe"
    elif any(x in location for x in ['singapore', 'mumbai', 'tokyo', 'shanghai', 'china', 'india', 'japan']):
        return "Asia"
    elif any(x in location for x in ['sao paulo', 'brazil', 'south america']):
        return "South America"
    elif any(x in location for x in ['dubai', 'uae', 'middle east']):
        return "Middle East"
    else:
        return "Other"

def engineer_features(df):
    """
    Adds new features to the dataframe for better risk analysis.
    """
    df = df.copy()
    if 'location' in df.columns:
        df['region'] = df['location'].apply(categorize_region)
    else:
        df['region'] = 'Unknown'
    if 'risk_score' in df.columns and 'region' in df.columns:
        geo_risk = df.groupby('region')['risk_score'].transform('mean')
        df['geographic_risk_index'] = geo_risk
    if 'avg_delay_days' in df.columns:
        max_delay = df['avg_delay_days'].max()
        min_delay = df['avg_delay_days'].min()
        ptp_delay = max_delay - min_delay
        if ptp_delay == 0: ptp_delay = 1
        df['normalized_delay'] = (df['avg_delay_days'] - min_delay) / ptp_delay

    if 'risk_score' in df.columns:
        max_risk = df['risk_score'].max()
        min_risk = df['risk_score'].min()
        ptp_risk = max_risk - min_risk
        if ptp_risk == 0: ptp_risk = 1
        df['normalized_risk'] = (df['risk_score'] - min_risk) / ptp_risk
    if 'normalized_delay' in df.columns and 'normalized_risk' in df.columns:
        df['composite_risk'] = 0.6 * df['normalized_delay'] + 0.4 * df['normalized_risk']
        df['supplier_tier'] = pd.cut(
            df['composite_risk'], 
            bins=[-float('inf'), 0.3, 0.7, float('inf')], 
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
    if 'total_deliveries' in df.columns and 'on_time_deliveries' in df.columns:
        df['on_time_delivery_rate'] = df['on_time_deliveries'] / df['total_deliveries']

    if 'total_units_shipped' in df.columns and 'defective_units' in df.columns:
        df['quality_defect_rate'] = df['defective_units'] / df['total_units_shipped']

    return df
