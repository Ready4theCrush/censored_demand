# censored_demand

This package simulates censored demand over a series of days and time periods
within each day. It then uses known intraday demand to predict total demand
of stockout days.

The simulation and analysis code is in:
`src/censored_demand/simulate.py`
`src/censored_demand/predict.py`

A demonstration of its use is in:
`notebooks/1.0_TH_predict_demand.ipynb`

## Simulation
First, an intraday demand curve is created to model real product
demand that might peak during certain parts of the day.

**generate_intraday_demand_curve**  
Given a number of time periods in the day, which is assumed to last from
0 to 11 hours, and at least one peak demand time in the day, this adds Gaussian
functions to create a normalized demand curve.
```python
# Create a day with 4 periods, and a peak early in the day.
demand_curve = generate_intraday_demand_curve(time_periods=4, peaks=[3])

# Gives:
# demand_curve: array([0.30897691, 0.49699262, 0.17947872, 0.01455175])
```

**predict_stockout_day_demand**  
This is the top level simulator of sales where demand
can possibly outstrip supply. Total demand is chosen randomly from a grv
 centered around `demand_mean` with standard deviation `demand_std`. Then, 
 the total demand is projected onto each time period according to the demand curve.
 Finally, demand during each time period is a Poisson process with expected value
 at the intraday mean.
 
 So the assumption is that demand has a certain basic shape throughout the day,
  but some days are busier than others, and that demand is random during each period.
  
In addition, total production (amount of product available for sale that day) is a
Gaussian random variable with `production_mean` and `production_std`. *Unless*, 
`fixed_production` kwarg is set to `True`, in which case production is fixed at
`production_mean` for every day. Sometimes we want to model production as something
don't know, but sometimes we want to see sales and waste numbers when production is fixed at
 at a specific level.

```python
intraday_sales_daily, unsold_daily = generate_daily_period_sales(
    demand_curve,
    days=6,
    demand_mean=10,
    demand_std=4,
    production_mean=8,
    production_std=3,
    fixed_production=False) 

# intraday_sales_daily:
# array([[4., 1., 0., 0.],
#        [4., 3., 0., 0.],
#        [4., 3., 0., 0.],
#        [1., 4., 1., 0.],
#        [0., 1., 0., 0.],
#        [2., 4., 0., 0.]])

# unsold_daily:
# array([0., 0., 0., 6., 4., 0.])
```

## Prediction

Total demand isn't known on stockout days, but it is known on days when there is
unsold product. These functions
 1.) Decide which days have known total demand.
 2.) Use those days to model total demand as a function of partially known day's sales.
 3.) Estimates total demand on stockout days.
 
 The assumption is that demand has a certain shape, and that on busy days, a good
 estimate of demand later in the day is an extension of the demand level seen so far
 for the rest of the day. 
 
 **split_days_by_stockout**  
 Use unsold product to decide which days have known demand,
  and which had a stockout.
 ```python
 # Divide up days into fully known demand and stockout days
known_demand_days, stockout_days = split_days_by_stockout(
    intraday_sales_daily,
    unsold_daily)
```

**create_models_by_known_periods**  
For a day with `T` Time periods, this builds `T-1` linear models for total demand
based on some subset `t` of known time periods. 

This automates the calculation we already do in our heads to approximate 
total demand: If you ran a coffee shop and ran out of milk at noon, you
would know that you had enough for the morning rush, but missed the afternoon
rush. Factoring in how much milk you started with, you could approximate how
 much more coffee you could have sold if you hadn't run out of milk.
 
```python
# Train models for each number of known demand periods
period_models = create_models_by_known_periods(known_demand_days)

# period_models:
# {1: <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f8564173700>,
#  2: <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f85373ebf70>,
#  3: <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f85373eb8e0>}
```

**predict_stockout_daily_demand**  
Makes predictions based on known demand before stockout.

```python
pred_stockout_demands = predict_stockout_daily_demand(stockout_days, period_models)

# pred_stockout_demands:
# array([[ 8.61261261],
#        [ 4.30630631],
#        [ 4.30630631]])
```

## Installation

In order to set up the necessary environment:

1. create an environment `censored_demand` with the help of [conda],
   ```
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```
   conda activate censored_demand
   ```
3. install `censored_demand` with:
   ```
   python setup.py install # or `develop`
   ```

## Project Organization

```
├── AUTHORS.rst             <- List of developers and maintainers.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── environment.yaml        <- The conda environment file for reproducibility.
├── notebooks               <- Jupyter notebooks. 
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── censored_demand     <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
```

## Note

This project has been set up using PyScaffold 3.2.3 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
