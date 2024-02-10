# -*- coding: utf-8 -*-
# pylint: disable=E1101,R0913,R0914,R0902
"""
Topological Data Anaylsis for Financial Time Series Data
========================================================
*Created on 23.12.2023 by bari_is*
*Copyright (C) 2023*
*For COPYING and LICENSE details, please refer to the LICENSE file*

This repository hosts the Python package developed from the research presented in the paper 
"Topological Tail Dependence: Evidence from Forecasting Realized Volatility" 
(https://doi.org/10.1016/j.jfds.2023.100107) by Hugo Gobato Souto. The package is designed to implement 
the methodologies and techniques described in the paper, focusing on the application of topological data 
analysis to understand tail dependence in financial markets.

"""
import warnings
from typing import List, Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import persim
from ripser import Rips
from tqdm import tqdm

from fintda.auxiliary import get_data_range

__all__ = ['FinTDA']


class FinTDA:
    """
    Central to finance-tda package is the FinTDA class, which encapsulates the computation of persistence diagrams to 
    analyze the tail dependence in financial time series data.

    Parameters
    ----------
    returns : pd.DataFrame
        A Pandas DataFrame containing asset return data, structured as NxM where N represents time 
        frames and M represents specific assets.
    weights : Sequence, optional
        The weights corresponding to the assets in the portfolio. If None, then the weights will be equal to 1/n,
        where n is the number of assets in the portfolio. Default is None. If the sum of the weights is not equal to 1,
        then the weights will be normalized to sum to 1.
    maxdim: int, optional
        An Integer parameter, borrowed from the ripser Python package, which specifies the highest dimension of 
        homology to compute, where higher dimensions represent more complex shapes in data. 
        It defaults to 2, indicating both H0 and H1 will be computed.
    thresh: float, optional
        A float value which determines the maximum edge length for simplices in the Vietoris-Rips filtration, 
        controlling the scale of topological features analyzed. The default value is set to infinity, which means, 
        that the entire filtration will be computed.
    coeff: int, optional
        An integer value that sets the prime field for homology calculations, affecting the algebraic structure of 
        the computation. Defaults to 2.

    Attributes
    ----------
    returns : pd.DataFrame
        The returns data for the financial time series.
    weights : Union[List, np.ndarray]
        The weights corresponding to the assets in the portfolio.
    _raw_data : pd.DataFrame
        The raw returns data.
    n : int
        The number of data points in the returns data.
    __max_date : pd.Timestamp
        The maximum date in the returns data.
    _portfolio_volatility : float
        The volatility of the portfolio.
    _mean_pnl : float
        The mean profit and loss (PnL) of the returns data.
    _volatility : float
        The volatility of the returns data.
    info : dict
        Additional information about the fintda instance.

    Methods
    -------
    compute_dgm(days=None, begin_date=None, end_date=None, plot=False)
        The compute_dgm method is designed to compute the persistence diagram of a given financial time 
        series data. This method applies topological data analysis (TDA) to uncover the underlying topological 
        features of the dataset, such as loops and voids, which persist over various scales. 
    compute_moving_dgm(windows_size=20, time_segments=None, days=None, begin_date=None, end_date=None, distance_method='wasserstein', plot=False)
        Compute the moving persistence diagram for a given time series.

    See Also
    --------
    [Hugo Gobato Souto](https://github.com/hugogobato/Topological-Tail-Dependence-Evidence-from-Forecasting-Realized-Volatility/blob/main/Persistent%20Homology%20for%20Realized%20Volatility%20(PH-RV)%20algorithm.py)

    """

    def __init__(self,
                 returns: pd.DataFrame,
                 weights: Optional[Sequence] = None,
                 maxdim: int = 2,
                 thresh: float = np.inf,
                 coeff: int = 2):

        if weights is None:
            weights = np.ones(returns.shape[1]) / returns.shape[1]

        if len(weights) != returns.shape[1]:
            raise ValueError("The length of the weights must be equal to the number of assets in the portfolio. "
                             "The current length of the weights is {len(weights)} and the number of assets in the portfolio is {returns.shape[1]}.")

        if np.sum(weights) != 1:
            weights = np.array(weights) / np.sum(weights)

        self.returns = returns * weights
        self.weights = np.atleast_1d(weights).flatten()

        self._raw_data = returns

        self.n = self.returns.index.shape[0]

        cov_matrix = self.returns.cov()

        self._portfolio_volatility = np.sqrt(self.weights.T.dot(cov_matrix).dot(self.weights))
        self._mean_pnl = np.mean(self.returns.values)
        self._volatility = np.std(self.returns.values)

        self.info = {
            "Mean PnL": self._mean_pnl,
            "Volatility": self._volatility,
            "Portfolio Volatility": self._portfolio_volatility
        }

        self._rips = Rips(maxdim=maxdim,
                          thresh=thresh,
                          coeff=coeff,
                          verbose=False)

    # ----------------------------------------------------------------------------------------------
    # Magic Methods
    # ----------------------------------------------------------------------------------------------
    def __repr__(self) -> str:
        """Return the string representation of the fintda instance.

        Returns
        -------
        str
            The string representation of the fintda instance.
        """
        head = "<VaR - {mu}: {mu_val}%, {sigma}: {sigma_val}%, " \
               "Portfolio {sigma}: {port_sigma_val}%>".format(mu=chr(956),
                                                              mu_val=round(self._mean_pnl * 100, 2),
                                                              sigma=chr(963),
                                                              sigma_val=round(self._volatility * 100, 4),
                                                              port_sigma_val=round(self._portfolio_volatility * 100, 4))

        return head

    # ----------------------------------------------------------------------------------------------
    # Public Methods
    # ----------------------------------------------------------------------------------------------

    def compute_dgm(self,
                    days: Optional[int] = None,
                    begin_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    plot: bool = False) -> List[np.ndarray]:
        """
        The compute_dgm method is designed to compute the persistence diagram of a given financial time series data.
        This method applies topological data analysis (TDA) to uncover the underlying topological features of 
        the dataset, such as loops and voids, which persist over various scales. 

        Parameters
        ----------
        days : Optional[int]
            An integer value that specifies the number of days for the data range to be analyzed, allowing 
            for a focused examination of specific time periods. Suppose you have a financial time series dataset 
            representing the daily returns of a portfolio over the past year. Instead of analyzing the entire year's 
            data, one can decide to focus on a 90-day period. If the parameter is None (default), the entire dataset 
            is considered unless the parameter begin_date and/or end_date are specified.
        begin_date : Optional[str]
            A string value of form 'YYYY-MM-DD', which sets the starting date of the analysis period, enabling 
            targeted analysis from a specific point in time. If both parameter days and begin_date are specified, 
            the data range will start from begin_date and span days days forward. If None, the data range starts 
            from the earliest date in the dataset.
        end_date : Optional[str]
            A string value of form 'YYYY-MM-DD', which sets the end date of the analysis period, enabling targeted 
            analysis from a specific point in time. If None, the data range ends at the latest date in the dataset.
        plot : bool, optional
            Whether to plot the persistence diagram.

        Returns
        -------
        List[np.ndarray]
            The method returns a collection of persistence diagrams, with one diagram for each dimension up to 
            the specified maximum dimension (the maxdim parameter). Each diagram is provided as a two-dimensional 
            numpy array, where the number of rows corresponds to the number of feature pairs identified in that 
            dimension. 

        Examples
        --------
        >>> from fintda import FinTDA
        >>> from fintda.auxiliary import load_data
        >>> # Determine some asset returns
        >>> returns = load_data()
        >>> ftda = FinTDA(returns)

        >>> # Compute the persistence diagram for the entire dataset
        >>> dgm = ftda.compute_dgm()

        >>> # Compute the persistence diagram for the first 90 days of data
        >>> dgm = ftda.compute_dgm(days=90)

        >>> # Compute the persistence diagram starting from a specific date
        >>> dgm = ftda.compute_dgm(begin_date='2022-01-01')

        >>> # Compute the persistence diagram starting from a specific date and ending 90 days later
        >>> dgm = ftda.compute_dgm(days=90, begin_date='2022-01-01')

        >>> # Compute the persistence diagram for a specific date range
        >>> dgm = ftda.compute_dgm(begin_date='2022-01-01', end_date='2022-03-31')

        >>> # Compute and plot the persistence diagram for the entire dataset
        >>> dgm = ftda.compute_dgm(plot=True)
        """

        data = get_data_range(self.returns, begin_date, end_date, days)
        dgm = self._rips.fit_transform(data.values)

        if plot:
            persim.plot_diagrams(dgm)

        return dgm

    def compute_moving_dgm(self,
                           windows_size: int = 20,
                           time_segments: Optional[int] = None,
                           days: Optional[int] = None,
                           begin_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           distance_method: Literal['wasserstein', 'bottleneck'] = 'wasserstein',
                           plot: bool = False) -> pd.Series:
        """
        The compute_moving_dgm method extends the capabilities of compute_dgm by applying TDA across moving windows 
        of the financial time series data. This approach allows for the dynamic analysis of topological features over 
        time, providing a deeper understanding of how the data's topological structure evolves.

        Parameters
        ----------
        windows_size : int, optional
            An integer value that defines the size of each moving window, controlling the granularity of the temporal 
            analysis. Default is 20.
        time_segments : int, optional
            An integer value, which specifies the number of segments for analysis, offering flexibility in the division 
            of the time series. If None (default), the time series will be divided into 
            n - (2 * windows_size) + 1 segments, where n is the length of the time series.
        days : Optional[int]
            An integer value that specifies the number of days for the data range to be analyzed, allowing 
            for a focused examination of specific time periods. Suppose you have a financial time series dataset 
            representing the daily returns of a portfolio over the past year. Instead of analyzing the entire year's 
            data, one can decide to focus on a 90-day period. If the parameter is None (default), the entire dataset 
            is considered unless the parameter begin_date and/or end_date are specified.
        begin_date : Optional[str]
            A string value of form 'YYYY-MM-DD', which sets the starting date of the analysis period, enabling 
            targeted analysis from a specific point in time. If both parameter days and begin_date are specified, 
            the data range will start from begin_date and span days days forward. If None, the data range starts 
            from the earliest date in the dataset.
        end_date : Optional[str]
            A string value of form 'YYYY-MM-DD', which sets the end date of the analysis period, enabling targeted 
            analysis from a specific point in time. If None, the data range ends at the latest date in the dataset.
        distance_method : {'wasserstein', 'bottleneck'}, optional
            The distance method to use for computing persistence diagrams. Default is 'wasserstein'.
        plot : bool, optional
            Whether to plot the computed distances. Default is False.

        Returns
        -------
        pd.Series
            The computed moving persistence diagram.

        Examples
        --------
        >>> from fintda import FinTDA
        >>> from fintda.auxiliary import load_data
        >>> # Determine some asset returns
        >>> returns = load_data()
        >>> ftda = FinTDA(returns)
        >>> moving_dgm = ftda.compute_moving_dgm(windows_size=30, plot=True)
        >>> print(moving_dgm)
        Date
        2022-01-01    0.1234
        2022-01-02    0.5678
        2022-01-03    0.9876
        ...
        dtype: float64

        References
        ----------
        [Hugo Gobato Souto](https://github.com/hugogobato/Topological-Tail-Dependence-Evidence-from-Forecasting-Realized-Volatility/blob/main/Persistent%20Homology%20for%20Realized%20Volatility%20(PH-RV)%20algorithm.py)

        Raises
        ------
        ValueError
            If the length of the data is less than 2 * windows_size.
        """
        # Environmental Variables ==========================================================
        data = get_data_range(self.returns, begin_date, end_date, days)

        # Check data length
        if len(data) < (2 * windows_size):
            raise ValueError(f"The length of the data is less than 2 * windows_size. Length of data: {len(data)}, "
                             f"Length of windows_size: {windows_size}")

        n = len(data) - (2 * windows_size) + 1 if time_segments is None else time_segments

        distance = np.zeros((n, 1))  # initialize array for distances

        distance_ufunc = persim.wasserstein if distance_method == 'wasserstein' else persim.bottleneck

        # Compute Distances ================================================================
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            # compute wasserstein distances between persistence diagrams for subsequent time windows
            for i in tqdm(range(n), desc="Computing Moving Diagrams"):

                # Compute persistence diagrams for adjacent time windows
                dgm1 = self._rips.fit_transform(data[i: i + windows_size].values)
                dgm2 = self._rips.fit_transform(data[i + windows_size + 1: i + (2 * windows_size) + 1].values)

                # Compute wasserstein distance between diagrams
                distance[i] = distance_ufunc(dgm1[0], dgm2[0], matching=False)

        # Compute Bands ====================================================================
        distance = pd.Series(distance.flatten(), index=data.index[windows_size:n+windows_size])

        if plot:
            distance_name = 'Wasserstein' if distance_method == 'wasserstein' else 'Bottleneck'

            # Create a figure and two subplots
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Plot 1: Portfolio
            ax1.plot(data.index[windows_size:n+windows_size], data.iloc[windows_size:n+windows_size, 0],
                     color='blue', linestyle='--', label='Portfolio (scaled)')

            ax1.set_ylabel('Returns')
            ax1.set_title('Portfolio Returns')

            # Plot 2: Distance
            ax2.plot(data.index[windows_size:n+windows_size], distance,
                     color='red', label=f"{distance_name} Distance (scaled)")

            ax2.set_xlabel('Date')
            ax2.set_ylabel('Topological Distance')
            ax2.set_title('Distance Changes')

            # Adjust spacing between subplots
            plt.tight_layout()

            # Display the plots
            plt.show()

        return distance
