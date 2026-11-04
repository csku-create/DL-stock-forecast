import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error
import math

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
    def accuracy_change(y_true_change, y_pred_change):
        """
        Accuracy for a price-change target.

        This is directional accuracy: the percentage of samples where the
        predicted and actual changes have the same sign. It avoids dividing
        by y_true_change, which is unstable when the actual change is near 0.
        """
        return EvaluationMetrics.directional_accuracy( y_true_change, y_pred_change )

    @staticmethod
    def get_tick_size(price, market):
        if market == "KL":
            if price < 1.00:
                return 0.005
            elif price < 10.00:
                return 0.01
            elif price < 100.00:
                return 0.02
            else:
                return 0.10
        elif market == "US":
            if price < 1.00:
                return 0.005
            else:
                return 0.01
        else:
            return 0.005

    @staticmethod
    def get_shares_per_trade(PI_types, market, price, capital, position_fraction):
        if PI_types in (1, 2):
            if market=="KL" :
                shares_per_trade = 100
            elif market=="US" :
                shares_per_trade = 10
            else:
                shares_per_trade = 1
        elif PI_types in (3, 4, 5, 6):
            if market=="KL" :
                shares_per_trade = int(((capital*position_fraction)/ price) // 100) * 100
            elif market=="US" :
                shares_per_trade = int((capital*position_fraction)/ price)
            else:
                shares_per_trade = 1
                
        return shares_per_trade

    @staticmethod #return 1 means trade, 0 means no trade    
    def check_should_sell(PI_types, trade_signal):
        if  PI_types in (1, 3, 5, 6):
            # Only trade if predicted price movement from today's close
            if trade_signal < 0:
                return 1
            else:
                return 0
        elif PI_types in (2, 4):
            # Sell exactly
            return 1        
        return 0

    @staticmethod  #malaysia cost assumption: brokerage_rate 0.10% RM8 minimum, clearing_rate 0.03% RM1,000 cap, stamp RM1.50 per RM1,000 block    
    def compute_trade_values(PI_types, market, trade_types, price, shares,
        us_commission_rate=0.0000, us_slippage_rate=0.0005, us_spread_rate=0.0010,
        my_brokerage_rate=0.0010,  my_brokerage_min=8.00, my_clearing_rate=0.0003, my_clearing_max=1000.00, my_stamp_per_1000=1.50, my_stamp_max=1000.00, my_slippage_rate=0.0010, my_spread_rate=0.0020
    ):       
        market = market.upper()
        if market in ("KL", "MY", "MALAYSIA", "BURSA"):
            market = "KL"
        elif market == "US":
            market = "US"
        else:
            raise ValueError("market must be 'US' or a Malaysian market identifier such as 'KL', 'MY', 'MALAYSIA', or 'BURSA'." )

        if market=="KL":
            if trade_types=="buy":
                execution_price = price * ( 1 + my_slippage_rate + my_spread_rate / 2 )
            elif trade_types=="sell":
                execution_price = price * ( 1 - my_slippage_rate - my_spread_rate / 2 )
            trade_value = shares * execution_price 

            brokerage = max( trade_value * my_brokerage_rate, my_brokerage_min )
            clearing_fee = min( trade_value * my_clearing_rate, my_clearing_max )
            stamp_blocks = math.ceil( trade_value / 1000.0 )    
            stamp_duty = min( stamp_blocks * my_stamp_per_1000, my_stamp_max )
            additional_cost = ( brokerage + clearing_fee + stamp_duty )
                      
        elif market=="US":
            if trade_types=="buy":
                execution_price = price *  ( 1 + us_slippage_rate + us_spread_rate / 2)
            elif trade_types=="sell":
                execution_price = price *  ( 1 - us_slippage_rate - us_spread_rate / 2)
            trade_value = shares * execution_price 
            additional_cost =  trade_value * us_commission_rate 
            
        else:
            trade_value = price * shares
            additional_cost = 0.00          
                
        if PI_types in (1, 2, 3, 4, 6):
            return price * shares
        elif PI_types==5:
            if trade_types=="buy":
                return  trade_value + additional_cost
            elif trade_types=="sell":
                return  trade_value - additional_cost
        else:
            return price * shares
        
    @staticmethod      
    def profitability_index(eval_types, y_pred, current_close, forecast_window, market, PI_types, initial_capital, position_fraction, buy_threshold=0):
        capital = float(initial_capital)       
        # if eval_types = price level :1=> y_pred: predicted next day value; current_close: actual observed close
        # if eval_types = price change :2 => y_pred: predicted next day change value; current_close: actual observed close
        # if eval_types = return : 3 => y_pred: predicted next day return value; current_close: actual observed close (not implement yet)
        countp=0
        counts=0
        y_pred = y_pred.flatten()
        current_close = current_close.flatten()

        if not PI_types==6:
            buy_threshold=0 #0.01 means 0.01%

        # Store all currently open purchases
        # Each item keeps the buy day and number of shares
        open_positions = []

        for i in range(len(y_pred)):            
            predicted_next_day = y_pred[i]
            # price level signal: positive = buy, negative = sell where predicted_next_day - today_price                         
            today_price = current_close[i]
            if eval_types == 1:
                trade_signal =  predicted_next_day - today_price
            elif eval_types == 2 or eval_types == 3:
                trade_signal = predicted_next_day
            
            # check position for sell
            positions_to_keep = []
            for position in open_positions:
                buy_day = position["buy_day"]
                shares = position["shares"]
                # Only allowed to sell after forecast_window days
                if i >= buy_day + forecast_window:
                    if  EvaluationMetrics.check_should_sell(PI_types, trade_signal)==1:                        
                        # Should Sell                                               
                        capital += EvaluationMetrics.compute_trade_values(PI_types=PI_types, market=market, trade_types="sell", price=today_price, shares=shares)
                        counts += 1
                        continue

                # Still holding
                positions_to_keep.append(position)

            open_positions = positions_to_keep

            # Only buy if there are enough remaining funds
            if i < len(y_pred) - forecast_window:
                shares_per_trade = EvaluationMetrics.get_shares_per_trade(PI_types, market, today_price, capital, position_fraction)
                if shares_per_trade <= 0:
                    continue
                buy_cost = EvaluationMetrics.compute_trade_values(PI_types=PI_types, market=market, trade_types="buy", price=today_price, shares=shares_per_trade)
                if trade_signal > 0 and capital >= buy_cost and (trade_signal/today_price*100) >buy_threshold:
                #if trade_signal >= EvaluationMetrics.get_tick_size(today_price, market) and capital >= buy_cost: # exceeds one market tick
                    capital -= buy_cost
                    countp += 1
                    open_positions.append({ "buy_day": i, "shares": shares_per_trade })
        
        # Sell any shares still being held at final actual price
        final_price = current_close[-1]
        for position in open_positions:
            capital += EvaluationMetrics.compute_trade_values(PI_types=PI_types, market=market, trade_types="sell", price=final_price, shares=position["shares"])
            counts += 1

        return round(capital / initial_capital, 4), countp, counts

        
                        
                