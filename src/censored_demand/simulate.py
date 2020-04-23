import logging
from typing import List, Dict, Tuple, Union, Any

import numpy as np
from scipy import stats
import statsmodels.api as sm

rgen = np.random.default_rng()

# Simulation methods
def generate_intraday_demand_curve(time_periods:int=12, peaks:List=[3]) -> np.ndarray:
    """Simulates a varying demand curve over time periods.

    Sums one or more Gaussian pdfs over a time period from 0 to 11 "hours"
    to create a simulated demand curve for use in simulating varying
    intraday demand.

    Args:
        time_periods (int): The number of periods in which to split
            a twelve hour period from hour 0 to 11.
        peaks (List): The hours at which to put the peaks of the demand curve.
            [1] would put the peak at the first hour
            [1, 9] would create a bimodal curve

    Returns:
        normalized_demand (np.ndarray): A sequence of float values
            normalized such that the sum is equal to one with
            shape = (time_periods,).
    """
    bins = np.linspace(0, 11, time_periods)
    
    #for each peak, create a gaussian pdf function with that peak hour as the mean
    raw_demands = [stats.norm(loc=peak, scale=3).pdf(bins) for peak in peaks]
    
    # add the discrete functions together and normalize
    raw_demand = np.sum(np.array(raw_demands), axis=0)
    normalized_demand = raw_demand / np.sum(raw_demand)
    return normalized_demand

def generate_daily_demand(
        intraday_demand_curve:np.ndarray,
        days:int=5,
        demand_mean:float=100,
        demand_std:float=5) -> np.ndarray:
    """Simulates intraday demand over a specified number of days.

    Given an intraday demand curve, this assumes each day has a random
    total demand chosen from a gaussian distribution, with mean = demand_mean
    and std = demand_std. Given this total demand each time period is
    assigned a mean period demand. Demand for products during each
    period is a poisson process with lambda equal to the mean period demand.
    Essentially, we choose how 'busy' the day is, then simulate intraday sales.

    Args:
        intraday_demand_curve (np.ndarray): normalized sequence of relative
            demands during each time period
        days (int): Number of days to simulate
        demand_mean (float): Mean for Gaussian random variable deciding
            overall busyness, or demand for the days
        demand_std (float): standard deviation for randomly chosen overall demand per day

    Returns:
        daily_period_demand (np.ndarray): Simulated demand (number of products desired
            for purchase) num_days over each time period. Shape is (days, time_periods).
    """
    
    # Generate gaussian total demands for each day, and round negative values up to zero
    daily_total_demands = rgen.normal(loc=demand_mean, scale=demand_std, size=days)
    daily_total_demands = np.where(daily_total_demands < 0, 0, daily_total_demands)
    
    # Given each daily demand, multiply by intraday curve to find mean demand per each period
    num_daily_periods = len(intraday_demand_curve)
    daily_period_demand_means = daily_total_demands[:,np.newaxis].dot(intraday_demand_curve[np.newaxis,:])
    
    # Simulate demand for products by day and period with Poisson a process
    daily_period_demand = rgen.poisson(daily_period_demand_means)
    return daily_period_demand

def generate_daily_production(
        days:int=5,
        production_mean:float=100,
        production_std:float=5,
        fixed_production:bool=False) -> np.ndarray:
    """Simulates total production for each day.
    
    Draws the overall number of a products produced for possible
    sale from a Gaussian distribution for each day.

    Args:
        days (int): Number of days over which to simulate total available products.
        production_mean (float): Mean of Guassian random variable simulating total products.
        production_std (float): Standard deviation of GRV
        fixed_production (bool): flag to identify of production should not be modeled as
            random, but as a fixed value equal to the production_mean value.

    Returns:
        daily_production (np.ndarray): Simulated total number of products available per
            day. Shape is (days,).
    """
    if fixed_production:
        daily_production = np.full(days, np.rint(production_mean))
    else:
        daily_production = rgen.normal(loc=production_mean, scale=production_std, size=days)
    return np.rint(daily_production)

def generate_daily_period_sales(
        intraday_demand_curve:np.ndarray,
        days:int=5,
        demand_mean:float=100,
        demand_std:float=5,
        production_mean:float=100,
        production_std:float=5,
        fixed_production:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Simulates censored demand for each intraday period over many days.
    
    Given a demand curve, a mean demand, and mean production, this return an
    array of realized intraday sales given that possibly demand outstrips
    supply of the product. The number of waste products is also provided
    for each day. This simulates a realistic intraday report and waste report
    that may be provided for a product.

    Args:
        intraday_demand_curve (np.ndarray): 1D array defining the relative demand
            for the product throughout the day. Used to define the lambdas of the
            Poisson processes that generate actual demand numbers during each period.            
        days (int): Number of days over which to simulate censored demand.
        demand_mean (float): Mean of Guassian random variable simulating total daily demand.
        demand_std (float): Standad Deviation of demand GRV generating daily demand.
        production_mean (float): Mean of Guassian random variable simulating total
            products for sale daily.
        production_std (float): Standard deviation of GRV for daily products.
        fixed_production (bool): flag to identify of production should not be modeled as
            random, but as a fixed value equal to the production_mean value.

    Returns:
        total_period_sales (np.ndarray): The sales during each period given that
            demand may exceed demand. Shape is (days, time_periods)
        unsold_daily (np.ndarray): The unsold product at the end of the day.
            Shape is (days,).
            
    Note:
        This model assumes a certain amount of product is available to be sold
        during the day and is sold to the first customers who request it, therefore
        a stockout occurs and then remains through the rest of the day.
    """
    # Generate intraday demand for each day
    daily_period_demand = generate_daily_demand(
        intraday_demand_curve,
        days=days,
        demand_mean=demand_mean,
        demand_std=demand_std)
    
    # Generate total demand for each day
    daily_production = generate_daily_production(
        days=days,
        production_mean=production_mean,
        production_std=production_std,
        fixed_production=fixed_production)
    
    # Find total demand for each day
    cumulative_period_demand = np.cumsum(daily_period_demand, axis=1)
    
    # Find how much product would remain for each period as demand
    # is subtracted from supply.
    remaining_by_period = daily_production[:,np.newaxis] - cumulative_period_demand
    
    # Find periods where remaining supply exceeded demand
    completed_period_sales = np.where(remaining_by_period < 0, 0, daily_period_demand)
    
    # Find periods where demand exceeded supply, but supply partially filled demand
    incompleted_sales = np.where(remaining_by_period < 0, daily_period_demand + remaining_by_period, 0)
    partial_period_sales = np.where(incompleted_sales < 0, 0, incompleted_sales)
    
    # Combine completed and incomplete sales to find total sales per period
    total_period_sales = completed_period_sales + partial_period_sales
    
    # Find total unsold product for each day
    total_daily_sales = np.sum(total_period_sales, axis=1)
    unsold_daily = daily_production - total_daily_sales

    return total_period_sales, unsold_daily