import pandas as pd
import streamlit as st

def run_eda(df):
    """Returns a dict with data quality info."""
    report = {
        'total_rows':     len(df),
        'duplicate_rows': df.duplicated().sum(),
        'missing_values': df.isnull().sum().to_dict(),
        'statistics':     df.describe().to_dict(),
    }

    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 1:
        report['correlation'] = df[numeric_cols].corr().to_dict()

    for col in ['avg_delay_days', 'risk_score']:
        if col in df.columns:
            report[f'{col}_skewness']  = df[col].skew()
            report[f'{col}_kurtosis'] = df[col].kurtosis()

    return report


def display_eda_section(df):
    """Renders the EDA report inside a Streamlit expander."""
    st.subheader("Data Quality Report (EDA)")

    with st.expander("Click to view"):
        report = run_eda(df)


        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", report['total_rows'])
        c2.metric("Duplicate Rows", report['duplicate_rows'])

        if any(report['missing_values'].values()):
            st.warning("Found missing values:")
            st.json(report['missing_values'])
        else:
            c3.success("No Missing Values")

        st.markdown("### Statistical Summary")
        st.dataframe(pd.DataFrame(report['statistics']))

        if 'correlation' in report:
            st.markdown("### Correlation Matrix")
            st.dataframe(pd.DataFrame(report['correlation']))
