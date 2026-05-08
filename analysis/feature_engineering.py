import pandas as pd


REGION_MAP = {
    'North America': ['usa', 'america', 'new york', 'los angeles', 'chicago'],
    'Europe':        ['london', 'berlin', 'paris', 'europe', 'uk', 'germany'],
    'Asia':          ['singapore', 'mumbai', 'tokyo', 'shanghai', 'china', 'india', 'japan'],
    'South America': ['sao paulo', 'brazil', 'south america'],
    'Middle East':   ['dubai', 'uae', 'middle east'],
}

def categorize_region(location):
    if not isinstance(location, str):
        return "Unknown"
    loc = location.lower()
    for region, keywords in REGION_MAP.items():
        if any(kw in loc for kw in keywords):
            return region
    return "Other"


def _normalize(series):
    """Min-Max normalize a column to 0–1 range."""
    range_ = series.max() - series.min()
    if range_ == 0:
        return series * 0  
    return (series - series.min()) / range_


def engineer_features(df):
    """Adds derived columns to the dataframe."""
    df = df.copy()

    
    if 'location' in df.columns:
        df['region'] = df['location'].apply(categorize_region)

   
    if 'risk_score' in df.columns and 'region' in df.columns:
        df['geographic_risk_index'] = df.groupby('region')['risk_score'].transform('mean')


    if 'avg_delay_days' in df.columns:
        df['normalized_delay'] = _normalize(df['avg_delay_days'])
    if 'risk_score' in df.columns:
        df['normalized_risk'] = _normalize(df['risk_score'])


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
