import pandas as pd
import streamlit as st

def run_eda(df):
    """
    Comprehensive EDA before agent analysis.
    Returns a dictionary of EDA results.
    """
    report = {}
    report['missing_values'] = df.isnull().sum().to_dict()
    report['duplicate_rows'] = df.duplicated().sum()
    report['total_rows'] = len(df)
    desc_cols = [col for col in ['avg_delay_days', 'risk_score'] if col in df.columns]
    if desc_cols:
         report['statistics'] = df[desc_cols].describe().to_dict()
    else:
        report['statistics'] = df.describe().to_dict()
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 1:
        report['correlation'] = df[numeric_cols].corr().to_dict()
    for col in ['avg_delay_days', 'risk_score']:
        if col in df.columns:
            report[f'{col}_skewness'] = df[col].skew()
            report[f'{col}_kurtosis'] = df[col].kurtosis()

    return report

def display_eda_section(df):
    """
    Helper function to display EDA in Streamlit.
    """
    st.subheader("📊 Data Quality Report (EDA)")

    with st.expander("Click to view detailed Data Quality Report"):
        eda_report = run_eda(df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", eda_report['total_rows'])
        c2.metric("Duplicate Rows", eda_report['duplicate_rows'])
        missing = eda_report['missing_values']
        if any(missing.values()):
            st.warning("Found missing values:")
            st.json(missing)
        else:
            c3.success("No Missing Values")

        st.markdown("### Statistical Summary")
        if 'statistics' in eda_report:
             st.dataframe(pd.DataFrame(eda_report['statistics']))

        if 'correlation' in eda_report:
            st.markdown("### Correlation Matrix")
            st.dataframe(pd.DataFrame(eda_report['correlation']))
