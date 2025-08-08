import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import os
from tensorflow.keras.models import load_model

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'stock_model.keras')
model = load_model(model_path)
st.header('Stock Market Predictor')

# Input from user
stock = st.text_input('Enter Stock Symbol (e.g. GOOG, AAPL)', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Fetch data
data = yf.download(stock, start, end)

# Show data
st.subheader('Stock Data')
st.write(data)

# Split data
data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test_full = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_full)

# Graph: MA50
st.subheader('Price vs MA50')
ma50 = data['Close'].rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma50, 'r', label='MA50')
plt.plot(data['Close'], 'g', label='Close Price')
plt.legend()
st.pyplot(fig1)

# Graph: MA50 & MA100
st.subheader('Price vs MA50 vs MA100')
ma100 = data['Close'].rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma50, 'r', label='MA50')
plt.plot(ma100, 'b', label='MA100')
plt.plot(data['Close'], 'g', label='Close Price')
plt.legend()
st.pyplot(fig2)

# Graph: MA100 & MA200
st.subheader('Price vs MA100 vs MA200')
ma200 = data['Close'].rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma100, 'r', label='MA100')
plt.plot(ma200, 'b', label='MA200')
plt.plot(data['Close'], 'g', label='Close Price')
plt.legend()
st.pyplot(fig3)

# Create test sequences
x = []
y = []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i, 0])
x = np.array(x)
y = np.array(y)

# Predict
predictions = model.predict(x)

# Rescale predictions
scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y = y * scale_factor

# Final graph
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, 'b', label='Original Price')
plt.plot(predictions, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)