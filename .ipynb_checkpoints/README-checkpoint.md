# censored_demand

This package simulates censored demand over a series of days and time periods
within each day. It then uses known intraday demand to predict total demand
of stockout days.

## Description

The actual simulation and analysis code is in:
`src/censored_demand/simulate.py`
`src/censored_demand/predict.py`

A demonstration of its use is in:
`/notebooks/1.0_TH_predict_demand.ipynb`

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
