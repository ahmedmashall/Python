import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from ttkthemes import ThemedTk
import pandas as pd
import numpy as np
import requests
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any
from urllib.parse import urlencode
from alpha_vantage.foreignexchange import ForeignExchange
from tradingview_ta import TA_Handler, Interval
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Set up caching
from joblib import Memory

memory = Memory(location='cache', verbose=0)


class ForexPredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Forex Prediction App')
        self.setup_ui()
        self.setup_data()
        self.setup_model()

    def setup_ui(self):
        # Set up main frame
        self.frame = ttk.Frame(self.master, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Set up currency pair dropdown menus
        self.currency_label1 = ttk.Label(self.frame, text='Select Base Currency:')
        self.currency_label1.grid(row=0, column=0, sticky=tk.W)
        self.currency_var1 = tk.StringVar(self.frame)
        self.currency_var1.set(major_currencies[0])
        self.currency_drop1 = ttk.OptionMenu(self.frame, self.currency_var1, *major_currencies)
        self.currency_drop1.grid(row=0, column=1, sticky=tk.W)

        self.currency_label2 = ttk.Label(self.frame, text='Select Quote Currency:')
        self.currency_label2.grid(row=1, column=0, sticky=tk.W)
        self.currency_var2 = tk.StringVar(self.frame)
        self.currency_var2.set(major_currencies[1])
        self.currency_drop2 = ttk.OptionMenu(self.frame, self.currency_var2, *major_currencies)
        self.currency_drop2.grid(row=1, column=1, sticky=tk.W)

        # Set up forecast interval dropdown menu
        self.interval_label = ttk.Label(self.frame, text='Select Forecast Interval:')
        self.interval_label.grid(row=2, column=0, sticky=tk.W)
        self.interval_var = tk.StringVar(self.frame)
        self.interval_var.set(forecast_intervals[0])
        self.interval_drop = ttk.OptionMenu(self.frame, self.interval_var, *forecast_intervals)
        self.interval_drop.grid(row=2, column=1, sticky=tk.W)

        # Set up trend direction radio buttons
        self.trend_label = ttk.Label(self.frame, text='Select Trend Direction:')
        self.trend_label.grid(row=3, column=0, sticky=tk.W)
        self.trend_var = tk.StringVar(self.frame)
        self.trend_var.set('Upward')
        self.trend_radio1 = ttk.Radiobutton(self.frame, text='Upward', variable=self.trend_var, value='Upward')
        self.trend_radio1.grid(row=3, column=1, sticky=tk.W)
        self.trend_radio2 = ttk.Radiobutton(self.frame, text='Downward', variable=self.trend_var, value='Downward')
        self.trend_radio2.grid(row=4, column=1, sticky=tk.W)

        # Set up prediction button
        self.predict_button = ttk.Button(self.frame, text='Predict', command=self.predict)
        self.predict_button.grid(row=5, column=1, sticky=tk.W)

        # Set up prediction results label
        self.prediction_label = ttk.Label(self.frame, text='')
        self.prediction_label.grid(row=6, column=0, columnspan=2, sticky=tk.W)

        # Set up progress bar
        self.progress_bar = ttk.Progressbar(self.frame, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
        self.progress_bar.grid(row=7, column=0, columnspan=2, sticky=tk.W)

    def setup_data(self):
        # Set up API key and foreign exchange connection
        self.api_key = '4OX66SPOKTG4WLB9'
        self.fx = ForeignExchange(key=self.api_key, output_format='pandas')

    def setup_model(self):
        # Set up neural network model
        self.model = Sequential(
            [
                LSTM(units=50, return_sequences=True, input_shape=(None, 1)),
                Dropout(0.2),
                BatchNormalization(),
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                BatchNormalization(),
                LSTM(units=50),
                Dropout(0.2),
                BatchNormalization(),
                Dense(units=1)
            ]
        )
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def predict(self):
        # Disable predict button and start progress bar
        self.predict_button.state(['disabled'])
        self.progress_bar.start()

        # Get selected currency pair and interval
        base_currency = self.currency_var1.get()
        quote_currency = self.currency_var2.get()
        interval = self.interval_var.get()

        # Set up API requests for forex data and TradingView technical analysis
        fx_url = f'https://api.exchangeratesapi.io/latest?base={base_currency}&symbols={quote_currency}'
        tv_handler = TA_Handler(
            symbol=f'{base_currency}{quote_currency}',
            screener="forex",
            exchange="FX_IDC"
        )

        try:
            # Fetch forex and technical analysis data
            fx_data, _ = self.fx.get_currency_exchange_intraday(from_symbol=base_currency, to_symbol=quote_currency,
                                                               interval=interval, outputsize='full')
            response = requests.get(tv_handler.get_analysis().summary)
            tv_data = json.loads(response.text)

            # Pre-process data
            fx_data = fx_data.rename(columns={'4. close': 'close'})
            fx_data['date'] = pd.to_datetime(fx_data.index)
            fx_data = fx_data[['date', 'close']]
            fx_data = fx_data.set_index('date')
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(fx_data['close'].values.reshape(-1, 1))
            x_train = []
            y_train = []
            for i in range(60, len(scaled_data)):
                x_train.append(scaled_data[i-60:i, 0])
                y_train.append(scaled_data[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Train model
            self.model.fit(x_train, y_train, epochs=20, batch_size=32)

            # Make prediction
            last_60_days = fx_data[-60:].values
            last_60_days_scaled = scaler.transform(last_60_days)
            X_test = []
            X_test.append(last_60_days_scaled)
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predicted_price = self.model.predict(X_test)
            predicted_price = scaler.inverse_transform(predicted_price)[0][0]

            # Set up prediction message
            if self.trend_var.get() == 'Upward':
                message = f'Based on the selected inputs, the next {interval} forecast for {base_currency}/{quote_currency} is predicted to increase to {predicted_price:.5f}.'
            else:
                message = f'Based on the selected inputs, the next {interval} forecast for {base_currency}/{quote_currency} is predicted to decrease to {predicted_price:.5f}.'

            # Display prediction message
            self.prediction_label.config(text=message)

        except Exception as e:
            # Display error message
            messagebox.showerror('Error', f'{e}')

        finally:
            # Enable predict button and stop progress bar
            self.predict_button.state(['!disabled'])
            self.progress_bar.stop()


# Set up major currency pairs and forecast intervals
major_currencies = ['USD', 'EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD']
forecast_intervals = ['1 Hour', '4 Hours', '1 Day']

# Set up caching
memory = Memory(location='cache', verbose=0)

# Run app
if __name__ == '__main__':
    root = ThemedTk(theme='arc')
    ForexPredictionApp(root)
    root.mainloop()
