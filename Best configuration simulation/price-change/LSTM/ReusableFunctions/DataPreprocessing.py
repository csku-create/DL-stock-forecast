import pandas as pd
import numpy as np
import yfinance as yf
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessing:
    def __init__(self, ticker=None, df=None, start_date='2014-01-01', end_date='2024-12-31'):
        self.analysis_start_date = '2015-01-01'
        if df is not None:
            self.df = df
        elif ticker is not None:
            self.df = self.download_stock_data(ticker, start_date, end_date)
            self.remove_exdividend_and_earnings_dates(ticker)
        else:
            raise ValueError("Either a ticker or a DataFrame must be provided.")

    def download_stock_data(self, ticker, start_date='2014-01-01', end_date='2024-12-31'):
        data = yf.download(ticker, start=start_date, end=end_date)

        # Clean before indicator calculation
        data.drop_duplicates(inplace=True)

        # Filter out negative prices or volumes
        price_cols = ['Open','High','Low','Close']
        for col in price_cols:
            data = data[data[col] >= 0]
        data = data[data['Volume'] >= 0]
        
        return data
        
    def remove_exdividend_and_earnings_dates(self, ticker):
        ticker_obj = yf.Ticker(ticker)

        # Get ex-dividend dates
        try:
            ex_dividends = ticker_obj.actions[ticker_obj.actions['Dividends'] > 0].index
            ex_dividends = ex_dividends.tz_localize(None).normalize()
        except Exception as e:
            print("Could not retrieve ex-dividend dates:", e)
            ex_dividends = pd.Index([])

        # Get earnings dates
        try:
            earnings_calendar = ticker_obj.earnings_dates
            earnings_dates = earnings_calendar.index
            earnings_dates = earnings_dates.tz_localize(None).normalize()
        except Exception as e:
            print("Could not retrieve earnings dates:", e)
            earnings_dates = pd.Index([])

        # Combine all dates
        combined_dates = ex_dividends.append(earnings_dates).drop_duplicates()

        # Normalize your dataframe index
        self.df.index = self.df.index.tz_localize(None).normalize()

        # Remove from DataFrame
        before = len(self.df)
        self.df = self.df[~self.df.index.isin(combined_dates)]
        after = len(self.df)

        print(f"Removed {before - after} rows corresponding to ex-dividend and earnings dates.")

    def add_technical_indicators(self):
        pd.options.mode.chained_assignment = None  # Disable warnings for chained assignments

        # Moving Averages
        self.df['20MA'] = self.df['Close'].rolling(window=20, min_periods=1).mean() #Takes sum of closing price of the last 20 days and divide by 20
        self.df['50MA'] = self.df['Close'].rolling(window=50, min_periods=1).mean() #Takes sum of closing price of the last 50 days and divide by 50

        # Relative Strength Index (RSI)
        delta = self.df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean() #Sum of positive values over 14 days
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean() #Sum of negative values over 14 days
        rs = gain / loss #Calculate the relative strength
        self.df['RSI'] = 100 - (100 / (1 + rs)) #>70 overbought, <30 oversold
        self.df['RSI'] = self.df['RSI'].fillna(50)

        # MACD and Signal Line
        self.df['12EMA'] = self.df['Close'].ewm(span=12, adjust=False, min_periods=1).mean() #EMA=Soothing Factor x Price + (1-Soothing Factor) x Previous EMA
        self.df['26EMA'] = self.df['Close'].ewm(span=26, adjust=False, min_periods=1).mean() #Soothing Factor= 2/(span)+1
        self.df['MACD'] = self.df['12EMA'] - self.df['26EMA']
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()

        # Bollinger Bands
        self.df['20STD'] = self.df['Close'].rolling(window=20, min_periods=1).std() #Standard deviation of Close over 20 periods
        self.df['Upper_BB'] = self.df['20MA'] + (self.df['20STD'] * 2)  # 20MA + 2 x Standard Deviation over a 20 week period
        self.df['Lower_BB'] = self.df['20MA'] - (self.df['20STD'] * 2)  # 20MA - 2 x Standard Deviation over a 20 week period

        # Commodity Channel Index (CCI)
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        mean_dev = lambda x: np.mean(np.abs(x - np.mean(x))) #sum of absolute value of previous typical price- typical price over a span of 20 weeks divide by the number of days
        self.df['CCI'] = (typical_price - typical_price.rolling(window=20, min_periods=1).mean()) / \
                        (0.015 * typical_price.rolling(window=20, min_periods=1).apply(mean_dev, raw=True)) #Typical Price -Typical Price Over a Span of 20 wewks/0.015 x Mean Diviation

        # Average True Range (ATR)
        self.df['TR'] = np.maximum(self.df['High'] - self.df['Low'], 
                                   np.maximum(abs(self.df['High'] - self.df['Close'].shift(1)), 
                                              abs(self.df['Low'] - self.df['Close'].shift(1))))
        self.df['ATR'] = self.df['TR'].rolling(window=14, min_periods=1).mean() #Simple Moving Average of TR Over a 14 Week Period
        self.df.drop(columns=['TR'], inplace=True)  # Drop intermediate column

        # Williams %R
        self.df['Williams_%R'] = ((self.df['High'].rolling(window=14, min_periods=1).max() - self.df['Close']) / 
                                  (self.df['High'].rolling(window=14, min_periods=1).max() - self.df['Low'].rolling(window=14, min_periods=1).min())) * -100   # (Highest High in the past 14 days - Close)/ (Highest High in the past 14 days-Lowest Low in the past 14 days) x -100

        # On-Balance Volume (OBV)
        self.df['OBV'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum() #Close > yesterday's, add todays volume to OBV ; Close < yesterday's, subtract todays volume from OBV

        # Drop intermediate columns
        self.df.drop(columns=['20STD'], inplace=True)

         #Forward-fill and backward-fill NaN values
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)

        # Verify no NaN values exist
        print("Null values in each column:\n", self.df.isnull().sum())
        print(f"Does the dataset contain any null values? {self.df.isnull().values.any()}")

        # Return only rows from analysis start date forward
        return self.df.loc[self.analysis_start_date:]

    def create_windowed_data(self, scaled_data, window_size: int = 50, forecast_window: int = 1):
        historicalinputwindow, next_price, current_price, next_price_change, next_price_return = [], [], [], [], []
        for i in range(window_size, len(scaled_data) - forecast_window + 1):            
            historicalinputwindow.append(scaled_data[i - window_size:i])
            current_close = scaled_data[i - 1][0]
            future_close = scaled_data[i + forecast_window - 1][0]
            
            next_price.append(future_close)  # assuming next day Close 
            current_price.append(current_close)  # assuming current day Close 
            next_price_change.append(future_close - current_close)  # assuming next day close - current day Close 
            next_price_return.append((future_close - current_close) / current_close)  # assuming next day return = (next day close - current day Close) / current day Close
        return np.array(historicalinputwindow), np.array(next_price), np.array(current_price), np.array(next_price_change), np.array(next_price_return)

    
    def split_dataset(self, X: np.ndarray, y: np.ndarray, z: np.ndarray, train_size: float = 0.7, val_size: float = 0.1):
        total_samples = len(X)
        train_end = int(total_samples * train_size)
        val_end = train_end + int(total_samples * val_size)
        
        return (
            X[:train_end], X[train_end:val_end], X[val_end:],
            y[:train_end], y[train_end:val_end], y[val_end:],
            z[:train_end], z[train_end:val_end], z[val_end:]
        )  

    
    #Only for train+val; test not included
    def prepare_data(self, selected_features, window_size, forecast_window):

        # Create windowed features and labels
        X, y = self.create_windowed_data(
            self.df[selected_features].values,
            window_size,
            forecast_window
        )

        # Split BEFORE scaling
        X_train, X_val, _, y_train, y_val, _ = self.split_dataset(X, y)

        # Feature scaling (fit on train, transform val)
        scaler_X = MinMaxScaler()
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train.shape)
        X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape)

        # Target scaling (fit on train, transform val)
        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

        return {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "y_train": y_train_scaled,
            "y_val": y_val_scaled,
            "scaler_y": scaler_y
        }

    def prepare_data_v2(self, selected_features, window_size, forecast_window):

        # Create windowed features and labels
        X, y = self.create_windowed_data_v2(
            self.df[selected_features].values,
            window_size,
            forecast_window
        )

        # Split BEFORE scaling
        X_train, X_val, _, y_train, y_val, _ = self.split_dataset(X, y)

        # Feature scaling (fit on train, transform val)
        scaler_X = MinMaxScaler()
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train.shape)
        X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape)

        # Target scaling (fit on train, transform val)
        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

        return {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "y_train": y_train_scaled,
            "y_val": y_val_scaled,
            "scaler_y": scaler_y
        }
    
    