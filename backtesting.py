def calculate_model_performance(df_historical):
    """
    Compare past predictions vs actual outcomes.
    Requires historical data with 'predicted_risky' and 'actually_caused_issue' columns.
    """
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    if 'actually_caused_issue' not in df_historical.columns or 'predicted_risk_probability' not in df_historical.columns:
        return {"error": "Missing validation columns"}
    y_true = df_historical['actually_caused_issue']
    y_pred = (df_historical['predicted_risk_probability'] > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    tn, fp, fn, tp = cm.ravel()

    if (tp + fn) > 0:
        detection_rate = tp / (tp + fn)
    else:
        detection_rate = 0

    return {
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'detection_rate': detection_rate,
        'missed_issues': fn,
        'false_alarms': fp
    }
