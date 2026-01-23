import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def train_predictive_model(df):
    """
    Trains a predictive model to forecast risk.
    Returns the dataframe with prediction probabilities and a feature importance dictionary.
    """
    df = df.copy()

    if 'caused_disruption' not in df.columns:
        df['predicted_risk_probability'] = 0.0
        return df, {}
    features = ['avg_delay_days', 'risk_score', 'geographic_risk_index', 'delay_volatility', 
                'lead_time_variance', 'on_time_delivery_rate']
    X = df[[col for col in features if col in df.columns]]

    if X.empty:
         df['predicted_risk_probability'] = 0.0
         return df, {}
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    y = df['caused_disruption']
    if len(df) < 5:
        df['predicted_risk_probability'] = 0.0
        return df, {}

    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_imputed, y)
        risk_probabilities = model.predict_proba(X_imputed)
        if risk_probabilities.shape[1] == 2:
            df['predicted_risk_probability'] = risk_probabilities[:, 1]
        else:
            df['predicted_risk_probability'] = 0.0
        importance = dict(zip(X.columns, model.feature_importances_))

    except Exception as e:
        print(f"Model training failed: {e}")
        df['predicted_risk_probability'] = 0.0
        importance = {}

    return df, importance
