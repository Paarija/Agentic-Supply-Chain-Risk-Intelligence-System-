import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_risk_dashboard(df):
    """Generate interactive risk visualizations."""

    charts = {}
    if 'risk_score' in df.columns:
        fig_hist = px.histogram(
            df, x='risk_score', 
            title='Risk Score Distribution',
            color_discrete_sequence=['#EF553B']
        )
        charts['histogram'] = fig_hist
    if 'avg_delay_days' in df.columns and 'risk_score' in df.columns:
        size_col = 'composite_risk_score' if 'composite_risk_score' in df.columns else None
        color_col = 'supplier_tier' if 'supplier_tier' in df.columns else None

        fig_scatter = px.scatter(
            df, x='avg_delay_days', y='risk_score',
            color=color_col,
            size=size_col,
            hover_data=['supplier_name'] if 'supplier_name' in df.columns else None,
            title='Delay vs Risk Analysis'
        )
        charts['scatter'] = fig_scatter
    if 'avg_delay_days' in df.columns and 'supplier_name' in df.columns:
        df_sorted = df.sort_values('avg_delay_days', ascending=False)
        if df_sorted['avg_delay_days'].sum() > 0:
            df_sorted['cumulative_pct'] = df_sorted['avg_delay_days'].cumsum() / df_sorted['avg_delay_days'].sum() * 100

            fig_pareto = go.Figure()
            fig_pareto.add_bar(x=df_sorted['supplier_name'], y=df_sorted['avg_delay_days'], name='Delay Days')
            fig_pareto.add_scatter(x=df_sorted['supplier_name'], y=df_sorted['cumulative_pct'], yaxis='y2', name='Cumulative %', mode='lines+markers')

            fig_pareto.update_layout(
                title='Pareto Analysis of Delays',
                yaxis=dict(title='Avg Delay Days'),
                yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 110])
            )

            charts['pareto'] = fig_pareto

    return charts
