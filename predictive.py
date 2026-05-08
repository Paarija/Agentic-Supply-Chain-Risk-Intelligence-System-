from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

def train_predictive_model(df):
    """Returns (df_with_predictions, feature_importance_dict)."""
    df = df.copy()
    default = (df.assign(predicted_risk_probability=0.0), {})

    if 'caused_disruption' not in df.columns or len(df) < 5:
        return default

    features = ['avg_delay_days', 'risk_score', 'geographic_risk_index',
                'delay_volatility', 'lead_time_variance', 'on_time_delivery_rate']
    available = [f for f in features if f in df.columns]
    if not available:
        return default

    X = df[available]
    y = df['caused_disruption']

    try:
        X_clean = SimpleImputer(strategy='mean').fit_transform(X)  

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_clean, y)

        probs = model.predict_proba(X_clean)
        df['predicted_risk_probability'] = probs[:, 1] if probs.shape[1] == 2 else 0.0

        importance = dict(zip(available, model.feature_importances_))
        return df, importance

    except Exception as e:
        print(f"Model training failed: {e}")
        return default
