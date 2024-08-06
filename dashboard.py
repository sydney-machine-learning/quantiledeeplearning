import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df

model_names = [
    'multi_bdlstm', 'uni_bdlstm', 'multi_edlstm', 'uni_edlstm', 'multi_clstm', 'uni_clstm'
]

datasets = {}

for model_name in model_names:
    try:
        datasets[model_name] = pd.read_csv(f'bitcoin_{model_name}_data.csv', parse_dates=['Date'])
        datasets[model_name] = clean_columns(datasets[model_name])
    except Exception as e:
        st.error(f"Error loading data for model {model_name}: {e}")

st.title('Cryptocurrency Close Price Dashboard: Bitcoin')

model_value = st.selectbox('Select Model', model_names)

date_ranges = {
    '2013': ['2013-01-01', '2013-12-31'],
    '2014': ['2014-01-01', '2014-12-31'],
    '2015': ['2015-01-01', '2015-12-31'],
    '2016': ['2016-01-01', '2016-12-31'],
    '2017': ['2017-01-01', '2017-12-31'],
    '2018': ['2018-01-01', '2018-12-31'],
    '2019': ['2019-01-01', '2019-12-31'],
    '2020': ['2020-01-01', '2020-12-31'],
    '2021': ['2021-01-01', '2021-12-31']
}

for key in date_ranges:
    date_ranges[key] = [pd.Timestamp(date) for date in date_ranges[key]]

selected_range = st.selectbox('Select Date Range', list(date_ranges.keys()))

start_date, end_date = date_ranges[selected_range]
filtered_df = datasets[model_value][(datasets[model_value]['Date'] >= start_date) & (datasets[model_value]['Date'] <= end_date)]

main_chart_df = filtered_df.copy()

filtered_df = filtered_df.dropna(subset=[f'Actual_{i}' for i in range(5)] + [f'Predicted_{i}' for i in range(5)])

fig = go.Figure()

fig.add_trace(go.Scatter(x=main_chart_df['Date'], y=main_chart_df['Close'], mode='lines', name='Close Price'))

fig.add_trace(go.Scatter(
    x=filtered_df['Date'], y=filtered_df['Close'], mode='markers', name='Prediction Point',
    marker=dict(color='rgba(0, 0, 255, 0.5)', size=8)
))

fig.update_layout(title='Close Price with Prediction Points', xaxis_title='Date', yaxis_title='Close Price')
st.plotly_chart(fig)

prediction_index = st.slider('Select Prediction Index', min_value=0, max_value=len(filtered_df) - 1, value=0)

row = filtered_df.iloc[prediction_index]

actual_values = [row[f'Actual_{i}'] for i in range(5)]
pred_values = [row[f'Predicted_{i}'] for i in range(5)]
quantile_05 = [row[f'Quantile_0.05_Timestep_{i}'] for i in range(5)]
quantile_25 = [row[f'Quantile_0.25_Timestep_{i}'] for i in range(5)]
quantile_50 = [row[f'Quantile_0.50_Timestep_{i}'] for i in range(5)]
quantile_75 = [row[f'Quantile_0.75_Timestep_{i}'] for i in range(5)]
quantile_95 = [row[f'Quantile_0.95_Timestep_{i}'] for i in range(5)]
close_dates = pd.date_range(start=row['Date'], periods=5)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=close_dates.tolist() + close_dates[::-1].tolist(),
    y=quantile_05 + quantile_25[::-1],
    fill='toself',
    fillcolor='rgba(0, 0, 255, 0.2)',
    line=dict(color='rgba(0, 0, 255, 0.5)'),
    showlegend=True,
    name='Quantile 0.05 - 0.25'
))
fig.add_trace(go.Scatter(
    x=close_dates.tolist() + close_dates[::-1].tolist(),
    y=quantile_25 + quantile_50[::-1],
    fill='toself',
    fillcolor='rgba(0, 255, 0, 0.2)',
    line=dict(color='rgba(0, 255, 0, 0.5)'),
    showlegend=True,
    name='Quantile 0.25 - 0.50'
))
fig.add_trace(go.Scatter(
    x=close_dates.tolist() + close_dates[::-1].tolist(),
    y=quantile_50 + quantile_75[::-1],
    fill='toself',
    fillcolor='rgba(255, 165, 0, 0.2)',
    line=dict(color='rgba(255, 165, 0, 0.5)'),
    showlegend=True,
    name='Quantile 0.50 - 0.75'
))
fig.add_trace(go.Scatter(
    x=close_dates.tolist() + close_dates[::-1].tolist(),
    y=quantile_75 + quantile_95[::-1],
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255, 0, 0, 0.5)'),
    showlegend=True,
    name='Quantile 0.75 - 0.95'
))

fig.add_trace(go.Scatter(x=close_dates, y=quantile_05, mode='lines', name='Quantile 0.05', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=close_dates, y=quantile_25, mode='lines', name='Quantile 0.25', line=dict(color='green')))
fig.add_trace(go.Scatter(x=close_dates, y=quantile_50, mode='lines', name='Quantile 0.50', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=close_dates, y=quantile_75, mode='lines', name='Quantile 0.75', line=dict(color='red')))
fig.add_trace(go.Scatter(x=close_dates, y=quantile_95, mode='lines', name='Quantile 0.95', line=dict(color='purple')))

fig.add_trace(go.Scatter(x=close_dates, y=actual_values, mode='lines+markers', name='Actual', line=dict(color='black')))
fig.add_trace(go.Scatter(x=close_dates, y=pred_values, mode='lines+markers', name='Predicted', line=dict(color='red', dash='dot')))

fig.update_layout(
    title=f'Prediction Detail for Index {prediction_index}',
    xaxis_title='Time Steps',
    yaxis_title='Values',
    legend=dict(x=0, y=1, traceorder='normal', bgcolor='rgba(0,0,0,0)'),
    plot_bgcolor='rgba(240, 240, 240, 1)',
    xaxis=dict(
        tickmode='array',
        tickvals=close_dates,
        ticktext=['1', '2', '3', '4', '5']
    ),
    height=600,
    margin=dict(l=50, r=50, t=50, b=50)
)

fig.update_traces(marker=dict(size=10))
st.plotly_chart(fig)
