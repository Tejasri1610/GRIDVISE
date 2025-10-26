# app.py - Refined for UDSA Lab Tasks 1-12 with interactive Streamlit UI

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import os

# ------------------- Streamlit Page Config -------------------
st.set_page_config(page_title="GridWise", layout="wide", initial_sidebar_state="expanded")

# ------------------- Design & Theme Settings -------------------
PRIMARY_COLOR = '#1F77B4'  # Blue
SECONDARY_COLOR = '#FFFFFF'  # White
st.markdown(f"<style>body{{background-color:{SECONDARY_COLOR}; color: #000;}}</style>", unsafe_allow_html=True)

# ------------------- Data Loading -------------------
@st.cache_data
def load_data(default_path=None):
    if default_path and os.path.exists(default_path):
        df = pd.read_csv(default_path)
        return df
    return None

# Parse datetime and create timestamp
def parse_datetime(df):
    df = df.copy()
    df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'])
    df['minutes_offset'] = (df['SETTLEMENT_PERIOD'] - 1) * 30
    df['timestamp'] = df.apply(lambda r: r['SETTLEMENT_DATE'] + pd.Timedelta(minutes=int(r['minutes_offset'])), axis=1)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# Prepare timeseries for forecasting
def prepare_timeseries(df, demand_col='ENGLAND_WALES_DEMAND'):
    df = parse_datetime(df)
    series = df[['timestamp', demand_col]].rename(columns={'timestamp': 'ds', demand_col: 'y'})
    series = series.set_index('ds').asfreq('30min')
    series['y'] = series['y'].interpolate(method='time')
    return series

# Create lag features
def create_lag_features(series, lags=[1,2,3,48,72,96]):
    df = series.copy().reset_index()
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['rolling_3'] = df['y'].shift(1).rolling(window=3).mean()
    df['rolling_48'] = df['y'].shift(1).rolling(window=48).mean()
    df = df.dropna().reset_index(drop=True)
    return df

# Train models
@st.cache_data
def train_model(X, y, model_type='RandomForest'):
    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'LinearRegression':
        model = LinearRegression()
    elif model_type == 'XGBoost':
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, objective='reg:squarederror')
    model.fit(X, y)
    return model

# Iterative forecast for multiple steps
def iterative_forecast(last_row, model, n_periods=48, lags=[1,2,3,48,72,96]):
    preds = []
    row = last_row.copy()
    for i in range(n_periods):
        X_pred = row.drop(labels=['ds','y']).to_frame().T
        yhat = model.predict(X_pred)[0]
        preds.append(yhat)
        # Update lag features
        for lag in sorted([c for c in row.index if c.startswith('lag_')], key=lambda x: int(x.split('_')[1])):
            lag_num = int(lag.split('_')[1])
            if lag_num == 1:
                row[lag] = yhat
            else:
                prev_col = f'lag_{lag_num-1}'
                row[lag] = row.get(prev_col, row[lag])
        row['rolling_3'] = (row.get('rolling_3', yhat) * 2 + yhat) / 3
        row['rolling_48'] = (row.get('rolling_48', yhat) * 47 + yhat) / 48
        row['y'] = yhat
    return preds

# ------------------- App Layout -------------------
st.title("GridWise — Smart Energy Demand Forecasting Dashboard")
st.markdown("**Explore UK half-hourly electricity demand & forecast with multiple models.**")

# Load default dataset
default_path = "/mnt/data/demanddata_2025 (1).csv"
df_default = load_data(default_path)

uploaded_file = st.sidebar.file_uploader("Upload demand CSV (or leave to use default)", type=['csv'])
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
else:
    if df_default is None:
        st.sidebar.warning("No default dataset found. Please upload your CSV.")
        st.stop()
    df_raw = df_default

if st.sidebar.checkbox("Show raw data preview"):
    st.subheader("Raw data (first 5 rows)")
    st.dataframe(df_raw.head())

# Demand column selection
demand_col = st.sidebar.selectbox("Select demand column", options=[c for c in df_raw.columns if 'DEMAND' in c.upper()], index=0)
series = prepare_timeseries(df_raw, demand_col=demand_col)

# Sidebar navigation
page = st.sidebar.radio("Page", ["Overview", "Forecast", "Renewables & Flows", "Discovery", "Feedback", "About"])
min_date = series.index.min().date()
max_date = series.index.max().date()
date_range = st.sidebar.date_input("Date range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(minutes=30)
series_filtered = series.loc[start_dt:end_dt]

# ------------------- Overview -------------------
if page == "Overview":
    st.header("Overview")
    col1, col2, col3 = st.columns(3)
    total_energy = series_filtered['y'].sum()
    peak = series_filtered['y'].max()
    peak_time = series_filtered['y'].idxmax()
    col1.metric("Total energy (sum over range)", f"{total_energy:,.0f} MW·periods")
    col2.metric("Peak demand", f"{peak:,.0f} MW", str(peak_time))

    # Renewable share
    wind_col = 'EMBEDDED_WIND_GENERATION' if 'EMBEDDED_WIND_GENERATION' in df_raw.columns else None
    solar_col = 'EMBEDDED_SOLAR_GENERATION' if 'EMBEDDED_SOLAR_GENERATION' in df_raw.columns else None
    if wind_col and solar_col:
        tmp = parse_datetime(df_raw).set_index('timestamp').loc[start_dt:end_dt]
        renewables_sum = tmp[wind_col].sum() + tmp[solar_col].sum()
        renew_pct = renewables_sum / (series_filtered['y'].sum()) * 100
        col3.metric("Renewable share (wind+solar)", f"{renew_pct:.2f}%")
    else:
        col3.write("Renewable data not available.")

    st.subheader("Demand trend")
    fig = px.line(series_filtered.reset_index(), x='ds', y='y', labels={'ds': 'Time', 'y': 'Demand (MW)'})
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Correlation matrix ----------------
    st.subheader("Correlation with numeric columns")
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    # Default columns to show in correlation
    default_corr = ['EMBEDDED_WIND_GENERATION','EMBEDDED_SOLAR_GENERATION','IFA_FLOW']
    default_corr = [c for c in default_corr if c in numeric_cols]
    corr_options = st.multiselect(
        "Choose columns to include in correlation",
        options=numeric_cols,
        default=default_corr if default_corr else numeric_cols[:5]
    )
    if corr_options:
        tmp_corr = parse_datetime(df_raw).set_index('timestamp').loc[start_dt:end_dt]
        corr_df = tmp_corr[corr_options + [demand_col]].corr()
        fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale='Blues',
                             title="Correlation Matrix (Selected Columns + Demand)")
        st.plotly_chart(fig_corr, use_container_width=True)


# ------------------- Forecast -------------------
elif page == "Forecast":
    st.header("Forecast")
    # Forecast settings
    lags = st.sidebar.multiselect("Select lag periods (30-min steps)", options=[1,2,3,4,5,6,24,48,72,96], default=[1,2,3,48,72,96])
    n_periods = st.sidebar.number_input("Forecast periods (30-min intervals)", min_value=1, max_value=96, value=48)
    model_choice = st.sidebar.selectbox("Choose model", options=["RandomForest", "LinearRegression", "XGBoost"], index=0)

    ts_df = series.reset_index().rename(columns={'ds':'ds','y':'y'})
    feat_df = create_lag_features(ts_df, lags=lags)

    test_size = st.sidebar.slider("Test size (%)", min_value=5, max_value=40, value=20)
    split_idx = int((1 - test_size/100) * len(feat_df))
    train = feat_df.iloc[:split_idx]
    test = feat_df.iloc[split_idx:]
    feature_cols = [c for c in feat_df.columns if c not in ['ds','y']]

    if st.button("Train & Forecast"):
        X_train, y_train = train[feature_cols], train['y']
        X_test, y_test = test[feature_cols], test['y']
        model = train_model(X_train, y_train, model_type=model_choice)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        st.success(f"{model_choice} Test MAE: {mae:,.2f} MW")

        # Actual vs Predicted plot
        recent = pd.DataFrame({'ds': test['ds'], 'Actual': y_test.values, 'Predicted': preds})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent['ds'], y=recent['Actual'], mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=recent['ds'], y=recent['Predicted'], mode='lines', name='Predicted', line=dict(color='orange')))
        st.plotly_chart(fig, use_container_width=True)

        # Iterative forecast
        last_row = feat_df.iloc[-1].copy()
        preds_forward = iterative_forecast(last_row, model, n_periods=n_periods, lags=lags)
        last_ts = ts_df['ds'].iloc[-1]
        future_index = [last_ts + pd.Timedelta(minutes=30*(i+1)) for i in range(n_periods)]
        forecast_df = pd.DataFrame({'Timestamp': future_index, 'Forecast (MW)': preds_forward})

        # Plot forecast
        fig2 = go.Figure()
        hist_slice = series.reset_index().tail(7*48)
        fig2.add_trace(go.Scatter(x=hist_slice['ds'], y=hist_slice['y'], mode='lines', name='History', line=dict(color='blue')))
        fig2.add_trace(go.Scatter(x=forecast_df['Timestamp'], y=forecast_df['Forecast (MW)'], mode='lines+markers', name='Forecast', line=dict(color='orange')))
        st.plotly_chart(fig2, use_container_width=True)

        # Show forecast table & download CSV
        st.subheader("Forecasted Values")
        st.dataframe(forecast_df.style.format({'Forecast (MW)': '{:,.2f}'}))
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Forecast CSV", data=csv, file_name='forecast_results.csv', mime='text/csv')

# ------------------- Renewables & Flows -------------------
elif page == "Renewables & Flows":
    st.header("Renewables & Interconnector Flows")
    tmp = parse_datetime(df_raw).set_index('timestamp').loc[start_dt:end_dt]
    # Renewable pie chart
    cols_for_pie = {}
    if 'EMBEDDED_WIND_GENERATION' in tmp.columns:
        cols_for_pie['Wind'] = tmp['EMBEDDED_WIND_GENERATION'].sum()
    if 'EMBEDDED_SOLAR_GENERATION' in tmp.columns:
        cols_for_pie['Solar'] = tmp['EMBEDDED_SOLAR_GENERATION'].sum()
    if len(cols_for_pie) > 0:
        pie_df = pd.DataFrame({'source': list(cols_for_pie.keys()), 'value': list(cols_for_pie.values())})
        fig = px.pie(pie_df, names='source', values='value', title='Renewable generation share')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No wind/solar data found.")

    # Interconnector flows
    flow_cols = [c for c in tmp.columns if 'FLOW' in c.upper() or '_FLOW' in c.upper()]
    if flow_cols:
        chosen = st.multiselect("Choose flow columns to plot", options=flow_cols, default=flow_cols[:3])
        if chosen:
            flow_df = tmp[chosen].reset_index()
            fig = px.line(flow_df, x='timestamp', y=chosen, labels={'timestamp':'Time'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No interconnector flow data.")

# ------------------- Discovery / Brainstorming (Task 1-3) -------------------
elif page == "Discovery":
    st.header("User Discovery & Brainstorm")
    st.markdown("**Share your insights, issues, or feature ideas:**")
    idea_input = st.text_area("Enter your thoughts here")
    if st.button("Submit Idea"):
        if idea_input.strip() != '':
            st.success("Thanks! Your idea has been captured.")
        else:
            st.warning("Please enter a valid idea.")

# ------------------- Feedback / Usability Testing (Task 10) -------------------
elif page == "Feedback":
    st.header("User Feedback")
    st.markdown("Rate the dashboard and provide your comments.")
    rating = st.slider("Overall experience", 1, 5, 4)
    comment = st.text_area("Additional comments")
    if st.button("Submit Feedback"):
        st.session_state['feedback'] = {'rating': rating, 'comment': comment}
        st.success("Feedback submitted (session only).")

# ------------------- About -------------------
elif page == "About":
    st.header("About GridWise")
    st.markdown("""
    **GridWise** is a student project — a Streamlit-based dashboard to explore UK half-hourly electricity demand,
    renewable generation, and interconnector flows. Includes multiple forecasting models (LinearRegression, RandomForest, XGBoost) with interactive visualization.
    """)
