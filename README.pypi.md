# Introduction

This repository hosts the Python package developed from the research presented in the paper "Topological Tail Dependence: Evidence from Forecasting Realized Volatility" by Hugo Gobato Souto. The package is designed to implement the methodologies and techniques described in the paper, focusing on the application of topological data analysis to understand tail dependence in financial markets.

The core of this package lies in its ability to model and forecast realized volatility in financial markets through the lens of topological data analysis. It presents a novel approach to understanding the complex relationships in financial data, especially during periods of high volatility or market stress.

# Installation

```cmd
> pip install fintda
```

You can also install the stable version with

```cmd

>>> pip install https://github.com/ibaris/finance-tda/archive/main.zip

```

To install the in-development version, change the branch name main to the other available branch names.

# Documentation

The documentation `code` documentation is in `build/docs`.

# Example

## Setup and Data Retrieval

Import Libraries: Import necessary libraries, including numpy, yfinance, and modules from the fintda package.

```python
import numpy as np
import yfinance as yf
from fintda import fintda
from fintda.auxiliary import get_data_range
%matplotlib inline
```

Define Financial Indexes and Date Range: Select the financial indexes and the date range for analysis.

```python
index_names = ['^GSPC', '^DJI', '^RUT']  # S&P 500, Dow Jones, Russell 2000
start_date_string = "2000-01-01"
end_date_string = "2022-03-30"
```

Retrieve Data from Yahoo Finance: Use yfinance to download historical data for the specified indexes and date range.

```python
raw_data = yf.download(index_names, start=start_date_string, end=end_date_string)
```

```console
[*********************100%%**********************]  3 of 3 completed
```

Data Preprocessing: Focus on adjusted closing prices and compute logarithmic returns.

```python
df_close = raw_data['Adj Close'].dropna(axis='rows')
returns = np.log(df_close.pct_change() + 1)
returns.dropna(inplace=True)
```

## Financial Time Series Analysis with fintda

Initialize fintda: Create an instance of fintda with the processed returns and predefined weights.

```python
weights = np.array([0.5, 0.3, 0.2])  # Define portfolio weights
ftda = fintda(returns, weights, log_returns=False)
```

```console
Rips(maxdim=2, thresh=inf, coeff=2, do_cocycles=False, n_perm = None, verbose=True)
```

Compute Moving Persistence Diagrams: Use the compute_moving_dgm method to calculate persistence diagrams. This method is essential for analyzing the topological features of the financial time series data.

```python
distance = ftda.compute_moving_dgm(plot=True)
```

```console
Computing Moving Diagrams: 100%|██████████| 5556/5556 [00:09<00:00, 580.01it/s]
```

# Reference

The development of this package is based on the research published in the following paper:

Souto, H.G. (2023). Topological Tail Dependence: Evidence from Forecasting Realized Volatility. The Journal of Finance and Data Science, 9. DOI: [10.1016/j.jfds.2023.100107](https://doi.org/10.1016/j.jfds.2023.10010)

The initial implementation from `hugogobato` can be found at:
[Topological Tail-Dependence Evidence](https://github.com/hugogobato/Topological-Tail-Dependence-Evidence-from-Forecasting-Realized-Volatility?tab=readme-ov-file)
