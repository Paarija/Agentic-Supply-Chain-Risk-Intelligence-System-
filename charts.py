import plotly.express as px
import plotly.graph_objects as go

def create_risk_dashboard(df):
    """Returns a dict of Plotly figures: histogram, scatter, pareto."""
    charts = {}

   
    if 'risk_score' in df.columns:
        charts['histogram'] = px.histogram(
            df, x='risk_score',
            title='Risk Score Distribution',
            color_discrete_sequence=['#EF553B']
        )

    
    if 'avg_delay_days' in df.columns and 'risk_score' in df.columns:
        charts['scatter'] = px.scatter(
            df, x='avg_delay_days', y='risk_score',
            color='supplier_tier'  if 'supplier_tier'       in df.columns else None,
            size='composite_risk_score' if 'composite_risk_score' in df.columns else None,
            hover_data=['supplier_name'] if 'supplier_name'  in df.columns else None,
            title='Delay vs Risk Analysis'
        )

    
    if 'avg_delay_days' in df.columns and 'supplier_name' in df.columns:
        sorted_df = df.sort_values('avg_delay_days', ascending=False)
        total = sorted_df['avg_delay_days'].sum()

        if total > 0:
            sorted_df['cumulative_pct'] = sorted_df['avg_delay_days'].cumsum() / total * 100

            fig = go.Figure()
            fig.add_bar(x=sorted_df['supplier_name'], y=sorted_df['avg_delay_days'], name='Delay Days')
            fig.add_scatter(x=sorted_df['supplier_name'], y=sorted_df['cumulative_pct'],
                            yaxis='y2', name='Cumulative %', mode='lines+markers')
            fig.update_layout(
                title='Pareto Analysis of Delays',
                yaxis=dict(title='Avg Delay Days'),
                yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 110])
            )
            charts['pareto'] = fig

    return charts
