import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, parse_dates=True)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, parse_dates=True)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        return df
    return None

def preprocess_data(df):
    # Try to detect date column
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if not date_cols:
        st.error("No date column found in the data.")
        return None, None
    date_col = date_cols[0]
    # Specify date format to avoid warning, assuming common format
    df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.rename(columns={date_col: 'Date'})

    # Try to detect sales and profit columns
    sales_cols = [col for col in df.columns if 'sales' in col.lower()]
    profit_cols = [col for col in df.columns if 'profit' in col.lower()]

    if not sales_cols:
        st.error("No sales column found in the data.")
        return None, None
    sales_col = sales_cols[0]

    if not profit_cols:
        # If no profit column, assume profit = 20% of sales
        df['Profit'] = df[sales_col] * 0.2
    else:
        profit_col = profit_cols[0]
        df['Profit'] = df[profit_col]

    df['Sales'] = df[sales_col]

    # Aggregate monthly
    df_monthly = df.groupby(pd.Grouper(key='Date', freq='M')).agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    df_monthly['Profit_Margin'] = df_monthly['Profit'] / df_monthly['Sales']
    return df_monthly, None

def calculate_metrics(df):
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    avg_profit_margin = df['Profit_Margin'].mean()
    return total_sales, total_profit, avg_profit_margin

def build_prophet_forecast(df):
    df_prophet = df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    # Use 3 months forecast horizon for performance balance
    future = model.make_future_dataframe(periods=3, freq='M')
    forecast = model.predict(future)
    return model, forecast

def plot_interactive_forecast(df, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Sales'], mode='lines+markers', name='Actual Sales'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Sales'))
    fig.update_layout(title='Interactive Sales Forecast',
                      xaxis_title='Date',
                      yaxis_title='Sales',
                      template='plotly_white')
    st.plotly_chart(fig)

def plot_profit_trends(df):
    if df is None:
        st.error("No data available to plot profit trends.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Profit'], mode='lines+markers', name='Profit'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Profit_Margin'], mode='lines+markers', name='Profit Margin'))
    fig.update_layout(title='Profit and Profit Margin Trends',
                      xaxis_title='Date',
                      yaxis_title='Value',
                      template='plotly_white')
    st.plotly_chart(fig)

def main():
    st.title("Retail Business Performance & Profitability Analysis")

    uploaded_file = st.file_uploader("Upload your retail data file (CSV or Excel)", type=['csv', 'xls', 'xlsx'])
    if uploaded_file is not None:
        if st.button("Generate Report"):
            st.session_state['generate_report'] = True
            # Removed st.experimental_rerun() due to AttributeError

    if st.session_state.get('generate_report', False):
        try:
            st.markdown("### Report Description")
            df_raw = load_data(uploaded_file)
            df = None
            if df_raw is not None:
                df, error = preprocess_data(df_raw)
            if df is not None:
                total_sales, total_profit, avg_profit_margin = calculate_metrics(df)
                description = (
                    f"This report analyzes your retail business performance based on the uploaded data. "
                    f"The total sales amount to ${total_sales:,.2f}, with a total profit of ${total_profit:,.2f}, "
                    f"resulting in an average profit margin of {avg_profit_margin:.2%}. "
                    "The sales forecast graph predicts future sales trends for the next 3 months, "
                    "while the profit trends graph shows historical profit and profit margin changes over time."
                )
            else:
                description = (
                    "This report provides an analysis of your retail business performance, "
                    "including key sales and profit metrics, sales forecasts, and profit trends. "
                    "Upload your retail data file in CSV or Excel format with date, sales, and profit columns "
                    "(profit is optional and assumed as 20% of sales if missing)."
                )
            st.markdown(description)
            progress_text = st.empty()
            progress_bar = st.progress(0)
            if df_raw is not None:
                progress_text.text("Loading data... 20%")
                progress_bar.progress(20)
                if df is not None:
                    progress_text.text("Calculating metrics... 50%")
                    progress_bar.progress(50)
                    total_sales, total_profit, avg_profit_margin = calculate_metrics(df)
                    st.subheader("Key Performance Metrics")
                    st.write(f"Total Sales: ${total_sales:,.2f}")
                    st.write(f"Total Profit: ${total_profit:,.2f}")
                    st.write(f"Average Profit Margin: {avg_profit_margin:.2%}")

                    # Add a small delay to allow UI update before heavy computation
                    import time
                    time.sleep(0.5)

                progress_text.text("Building forecast model... 75%")
                progress_bar.progress(75)
                model, forecast = build_prophet_forecast(df)
                st.subheader("Sales Forecast")
                if forecast is not None:
                    plot_interactive_forecast(df, forecast)
                else:
                    st.info("Forecast model is disabled for faster startup.")

                progress_text.text("Plotting profit trends... 100%")
                progress_bar.progress(100)
                st.subheader("Profit Trends")
                if df is not None:
                    plot_profit_trends(df)
                else:
                    st.error("No data available to plot profit trends.")
                progress_text.text("Report generation completed.")
        except Exception as e:
            st.error(f"An error occurred during report generation: {e}")
    else:
        st.info("Please upload a retail data file to begin analysis.")

if __name__ == "__main__":
    main()
