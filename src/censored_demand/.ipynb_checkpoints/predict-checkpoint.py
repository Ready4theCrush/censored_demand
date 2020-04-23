import logging
from typing import List, Dict, Tuple, Union, Any

import numpy as np
from scipy import stats
import statsmodels.api as sm


# Analysis methods

def split_days_by_stockout(
        period_sales_daily:np.ndarray,
        unsold_daily:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Splits intraday sales records into stockout and non-stockout (completed).
    
    From the unsold product at the end of the each day, this returns days
    assumed to be non-stockout ("complete"d sales days), and stockout days.

    Args:
        period_sales_daily (np.ndarray): Possibly censored sales
            numbers during each period of each day.
        unsold_daily (np.ndarray): Number of product units left
            at the end of each day.

    Returns:
        complete_days (np.ndarray): Days when product was left over,
            therefore all demand was met. Shape is (complete_days, time_periods).
        stockout_days (np.ndarray): Days when no waste was recorded,
            and stockout possibly occurred. Shape is (stockout_days, time_periods).
            
    Note:
        num_complete_days + num_stockout_days = days
    """
    complete_days = period_sales_daily[unsold_daily > 0]
    stockout_days = period_sales_daily[unsold_daily == 0]
    return complete_days, stockout_days

def create_models_by_known_periods(complete_days:np.ndarray) -> Dict:
    """Trains prediction models for each possible known periods scenario.
    
    To predict total actual demand on stockout days, total demand
    for some periods will be known, but not others. This method
    trains prediction models (OLS) for the total demand based on
    subsets of the total intraday demand. 

    Args:
        complete_days (np.ndarray): Intraday sales for days
            when stockout was known not to have occurred.
            
    Returns:
        period_models (Dict): Keys for this dictionary are the
            total number of periods in the day used to train
            the prediction model, the value is the actual
            trained statsmodels.OLS model. Keys are [1...time_period-1]
            
    Note:
        The assumption in this model is that total demand is a function
        of the partial demand earlier in the day, and that the prediction
        is more accurate, the more of the day you are able to use to make
        the prediction.
    """
    num_periods = complete_days.shape[1]
    period_models = {}
    for period in range(1,num_periods):
        # find subset of time_periods to train the model
        known_periods = complete_days[:,0:period]
        X = np.sum(known_periods, axis=1)
        
        # dependent variable is the total demand for the day
        Y = np.sum(complete_days, axis=1)
        ols_model = sm.OLS(Y,X)
        result = ols_model.fit()
        
        # Store the trained model where the key is the number of 
        # time periods used to train the model
        period_models[period] = result
    return period_models

def predict_stockout_day_demand(
        stockout_period_sales:np.ndarray,
        period_models:Dict) -> Any:
    """Predicts demand for a single row (day) where stockout occurred.
    
    Given a day when there was no waste, this assumes the only periods
    with known demand are the ones before the last period with a sale
    occurred. The known periods are then used to predict the total demand.

    Args:
        stockout_period_sales (np.ndarray): The intraday sales for one
            day where stockout possibly occurred.
        period_models (Dict): The dict with the trained models predicting
            total daily demand based on the "key" number of known demand
            time periods.
    
    Returns:
        predicted_days_demand (Union[np.ndarray, np.NaN]): The predicted
            total demand for the day. Shape is (1,).
            
    Example:
        Let's say intraday sales where: [23, 45, 23, 2, 0, 0, 0]
        In this case, only the first three periods are known to have supply
        exceed demand, so the number of known periods is 3. We use the sum
        of the demand from those periods as input to the prediction model stored 
        in period_models[3].
    """
    
    # Find which periods have known demand
    days_sum = np.sum(stockout_period_sales)
    cumsum_periods = np.cumsum(stockout_period_sales)
    known_periods = np.where(cumsum_periods < days_sum, stockout_period_sales, 0)
    num_known_periods = np.count_nonzero(known_periods)
    
    # Sum the known demand
    known_demand = np.sum(stockout_period_sales[0:num_known_periods])
    
    # predict total demand, but if there are no known demand periods, return np.NaN
    if num_known_periods > 0:
        predicted_days_demand = period_models[num_known_periods].predict(known_demand)
        return predicted_days_demand
    else:
        return np.NaN

def predict_stockout_daily_demand(
        stockout_days:np.ndarray,
        period_models:Dict) -> np.ndarray:
    """Finds predicted total demand for stockout days.
    
    Args:
        stockout_days (np.ndarray): The intraday sales for all
            days where stockout possibly occurred.
        period_models (Dict): The dict with the trained models predicting
            total daily demand based on the "key" number of known demand
            time periods.
    
    Returns:
        predicted (np.ndarray): The predicted
            total demand for the day. Shape is (num_stockout_days,1).
    """
    predicted = np.apply_along_axis(predict_stockout_day_demand, 1, stockout_days, period_models)
    return predicted