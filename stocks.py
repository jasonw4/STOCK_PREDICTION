import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

startDate = "2017-01-01"
currDate = date.today().strftime("%Y-%m-%d")

st.title("DOW JONES STOCK PREDICTOR")

stockList = ("AXP", "AMGN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "KO", "JPM", "MCD", "MMM", "MRK", "MSFT", "NKE", 
            "PG", "TRV", "UNH", "CRM", "VZ", "V", "WBA", "WMT", "DIS", "D")
selected = st.selectbox("Select DataSet for Prediction", stockList)

n_years = st.slider("# of Years Prediciting", 1, 10)
time_period = n_years*365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, startDate, currDate)
    data.reset_index(inplace=True)
    return data

data_loadstate = st.text("Load data...")
data = load_data(selected)
data_loadstate.text("Loading data...done...")

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    figure.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    figure.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

plot_raw_data()

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=time_period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1= plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
