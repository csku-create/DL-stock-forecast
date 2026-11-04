import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error

class EvaluationMetrics:
    @staticmethod
    def rmse(y_true, y_pred):
        return round(np.sqrt(mean_squared_error(y_true, y_pred)),4)

    @staticmethod
    def mape(y_true, y_pred):
        return round(mean_absolute_percentage_error(y_true, y_pred) * 100,4) # Not used in this research but useful metric

    @staticmethod
    def r2(y_true, y_pred):
        return round(r2_score(y_true, y_pred),4)
    
    @staticmethod
    def mae(y_true, y_pred):
        return round(mean_absolute_error(y_true, y_pred),4)

    @staticmethod
    def accuracy(y_true, y_pred, threshold_percent=5):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        y_true = np.where(y_true == 0, 1e-8, y_true)
        percentage_diff = np.abs((y_pred - y_true) / y_true) * 100
        return round((np.sum(percentage_diff <= threshold_percent) / len(y_true)) * 100,4)
    
    @staticmethod
    @staticmethod
    def profitability_index(y_true, y_pred, forecast_window, initial_capital=10000.0):
       capital = initial_capital
       position = 0
       y_true = y_true.flatten()
       y_pred = y_pred.flatten()

       i = 0
       while i < len(y_pred) - forecast_window:
           current_price = y_true[i]
           predicted_future_price = y_pred[i]

           # Buy if future price is predicted higher
           if predicted_future_price > current_price and position == 0:
               position = capital / current_price
               capital = 0
               # hold until forecast horizon
               i += forecast_window  

           # Sell if future price is predicted lower 
           elif predicted_future_price < current_price and position > 0:
               capital = position * current_price
               position = 0
               i += 1  

           else:
               i += 1  

       # Final liquidation if still holding
       if position > 0:
           capital = position * y_true[-1]

       return round(capital / initial_capital,4)

    @staticmethod
    def directional_accuracy(y_true, y_pred):
        """
        Percentage of predictions with the correct
        direction of price movement.

        Positive = price increase
        Negative = price decrease
        Zero     = no change
        """

        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        actual_direction = np.sign(y_true)
        predicted_direction = np.sign(y_pred)

        correct = (actual_direction == predicted_direction)

        return round(np.mean(correct) * 100, 4)

    @staticmethod
    def profitability_index_change( y_pred, close_price, forecast_window, initial_capital=10000.0):

        capital = initial_capital
        position = 0

        y_pred = np.asarray(y_pred).flatten()
        close_price = np.asarray(close_price).flatten()

        i = 0

        while i < len(y_pred) - forecast_window:

            current_price = close_price[i]
            future_price = close_price[i + forecast_window]

            # Buy if price is predicted to increase
            if y_pred[i] > 0:

                shares = capital / current_price

                # Buy now and liquidate at forecast horizon
                capital = shares * future_price

                i += forecast_window

            else:
                i += 1

        return round( capital / initial_capital, 4 )

    @staticmethod
    def profitability_index_v1(y_true, y_pred, forecast_window, initial_capital=100000.0):
        capital = initial_capital
        shares_per_trade = 100

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Store all currently open purchases
        # Each item keeps the buy day and number of shares
        open_positions = []

        for i in range(len(y_pred)):

            current_price = y_true[i]
            predicted_price = y_pred[i]
  
            # -------------------------------------------------
            # 1. CHECK EXISTING POSITIONS FOR SELL
            # -------------------------------------------------
            positions_to_keep = []

            for position in open_positions:

                buy_day = position["buy_day"]
                shares = position["shares"]

                # Only allowed to sell after forecast_window days
                if i >= buy_day + forecast_window:

                    # Sell signal
                    if predicted_price < current_price:
                        capital += shares * current_price
                        #print(f"SELL | day={i}, price={current_price:.2f}, " f"capital={capital:.2f}")
                        continue  # position has been sold
                        

                # Still holding
                positions_to_keep.append(position)

            open_positions = positions_to_keep

            # -------------------------------------------------
            # 2. CHECK FOR NEW BUY
            # -------------------------------------------------
            # Only buy if there are enough remaining funds
            if i < len(y_pred) - forecast_window:

                buy_cost = shares_per_trade * current_price

                if (predicted_price > current_price and capital >= buy_cost ):
                    capital -= buy_cost
                    #print( f"BUY  | day={i}, price={current_price:.2f}, " f"capital={capital:.2f}")
                    open_positions.append({ "buy_day": i, "shares": shares_per_trade })
                

        # -------------------------------------------------
        # 3. FINAL LIQUIDATION
        # -------------------------------------------------
        # Sell any shares still being held at final actual price
        final_price = y_true[-1]

        for position in open_positions:
            capital += position["shares"] * final_price

        return round(capital / initial_capital, 4)