import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# Set the page title and configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.header("Stock Watch")

# Set the sidebar
st.sidebar.title("Stock Price Prediction")
st.sidebar.success("Select a page above")

# Add an option for users to upload their own CSV file
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

# Function to load stock data
@st.cache
def load_stock_data(symbol):
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1=0&period2=9999999999&interval=1d&events=history&includeAdjustedClose=true"
    stock_data = pd.read_csv(url)
    return stock_data

    

forecast_days = 100

# Load the user-uploaded CSV file or ask for stock symbol
if uploaded_file is not None:
    stock_data = pd.read_csv(uploaded_file)
else:
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL for Apple):")
    if stock_symbol:
        # Button to trigger data loading
        load_button = st.sidebar.button("Load Data")
        if load_button:
            # Fetch the stock data using the stock symbol
            stock_data = load_stock_data(stock_symbol)
    else:
        st.sidebar.write("Please enter a stock symbol.")

        

# Display the stock data
if 'stock_data' in locals():
    st.subheader("Stock Data")
    st.write(stock_data.tail())
    st.write(np.asarray(stock_data).dtype)

    # Handle missing or NaN values
    stock_data = stock_data.dropna()

    # Remove commas from numerical values
    stock_data = stock_data.replace(",", "", regex=True)
    
    # Convert numerical columns to appropriate data types
    stock_data['Close'] = pd.to_numeric(stock_data['Close'])

    # Convert 'Close' column to float
    #stock_data['Close'] = stock_data['Close'].astype(float)

    # Perform stock price prediction
    df = stock_data.reset_index()
    df = df[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    model = ARIMA(df, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)

    st.write("New FORECAST")
    st.write(forecast)


    # Create future dates for the forecast
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=forecast_days)

    # Convert forecasted values to pandas DataFrame
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Close': forecast})

    # Set 'Date' column as index
    forecast_df.set_index('Date', inplace=True)

    # Combine actual data and forecasted values
    prediction_data = pd.concat([df['Close'], forecast_df['Forecasted Close']], axis=1)


    # Plot the forecasted stock prices
    st.subheader("Stock Price Forecast Chart using ARIMA")
    st.line_chart(prediction_data)

    model = SARIMAX(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)

    # Convert forecasted values to pandas DataFrame
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Close': forecast})

    # Set 'Date' column as index
    forecast_df.set_index('Date', inplace=True)

    # Combine actual data and forecasted values
    prediction_data_sarimax = pd.concat([df['Close'], forecast_df['Forecasted Close']], axis=1)

    # Plot the forecasted stock prices
    st.subheader("Stock Price Forecast Chart using SARIMAX")
    st.line_chart(prediction_data_sarimax)
    
    st.subheader('Future Improvements')

    st.write('''
    1. Advanced Model Selection: Explore and evaluate other advanced forecasting models such as LSTM (Long Short-Term Memory) neural networks or Prophet to compare their performance with ARIMA and SARIMAX models. This can help identify the most suitable model for accurate stock price predictions.
    2. Feature Engineering: Incorporate additional relevant features such as volume, news sentiment analysis, or technical indicators (e.g., moving averages, relative strength index) to enhance the predictive power of the models. Feature engineering can provide more meaningful insights and improve the accuracy of stock price forecasts.
    3. Hyperparameter Tuning: Optimize the hyperparameters of the ARIMA and SARIMAX models to achieve better forecasting results. This can be done through grid search or other techniques to find the optimal values for parameters like the order of differencing, autoregressive and moving average orders, and seasonal components.
    ''')

    st.subheader('Risks and Limitations')

    st.write('''
    1. Market Volatility: Stock markets can be highly volatile and subject to sudden changes due to various factors such as economic events, political developments, or market sentiment. These sudden shifts in market conditions can impact the accuracy of stock price predictions, making them inherently risky.
    2. Limited Historical Data: The accuracy of forecasting models heavily relies on the availability and quality of historical data. Insufficient or unreliable historical data can affect the performance of the models and result in less accurate predictions.
    3. Assumption of Stationarity: ARIMA and SARIMAX models assume stationarity, meaning that the statistical properties of the data remain constant over time. However, stock prices often exhibit non-stationary behavior, including trends, seasonality, and irregular fluctuations. Failure to adequately address non-stationarity can lead to inaccurate forecasts.
    4. External Factors: Stock prices can be influenced by external factors that are not captured in the historical data or the forecasting models. Factors like economic indicators, regulatory changes, or company-specific news can impact stock prices, making it challenging to accurately predict their movements solely based on historical patterns.
    Note: It's important to consider these limitations and risks while interpreting the results of stock price forecasting models and making investment decisions.
    ''')
