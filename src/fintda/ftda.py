# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
Topological Tail Dependence
===========================
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
from typing import List, Literal, Optional, Union

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
    fintda class for computing persistence diagrams and analyzing financial time series data.

    Parameters
    ----------
    returns : pd.DataFrame
        The returns data for the financial time series.
    weights : Union[List, np.ndarray]
        The weights corresponding to the assets in the portfolio.
    maxdim : int, optional
        The maximum dimension for computing persistence diagrams. Default is 2.
    **kwargs : dict
        Additional keyword arguments.

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
        Compute the persistence diagram for the given data.
    compute_moving_dgm(windows_size=20, time_segments=None, days=None, begin_date=None, end_date=None, distance_method='wasserstein', plot=False)
        Compute the moving persistence diagram for a given time series.

    See Also
    --------
    [Hugo Gobato Souto](https://github.com/hugogobato/Topological-Tail-Dependence-Evidence-from-Forecasting-Realized-Volatility/blob/main/Persistent%20Homology%20for%20Realized%20Volatility%20(PH-RV)%20algorithm.py)

    """

    def __init__(self,
                 returns: pd.DataFrame,
                 weights: Union[List, np.ndarray],
                 maxdim: int = 2,
                 **kwargs):

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

        self._rips = Rips(maxdim=maxdim)

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
        """Compute the persistence diagram (dgm) for the given data.

        Parameters
        ----------
        days : Optional[int]
            Number of days to consider for the data range (see Notes).
        begin_date : Optional[str]
            Start date of the data range in the format 'YYYY-MM-DD'. If None, then the data range will start 
            from the first date in the data.
        end_date : Optional[str]
            End date of the data range in the format 'YYYY-MM-DD'. If None, then the data range will end
            at the last date in the data.
        plot : bool, optional
            Whether to plot the persistence diagram.

        Returns
        -------
        List[np.ndarray]
            The computed persistence diagram.

        Notes
        -----
        If `days` and `begin_date` is specified, then the data range will be from `begin_date` to  
        `begin_date + days`.

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
        Compute the moving persistence diagram (dgm) for a given time series.

        Parameters
        ----------
        windows_size : int, optional
            The size of the moving window. Default is 20.
        time_segments : int, optional
            The number of time segments to divide the time series into. If None (default), then the time
            series will be divided into n - (2 * windows_size) + 1 segments, where n is the length of the
            time series.
        The end date of the data range in the format 'YYYY-MM-DD'.
        days : Optional[int]
            Number of days to consider for the data range (see Notes).
        begin_date : Optional[str]
            Start date of the data range in the format 'YYYY-MM-DD'. If None, then the data range will start 
            from the first date in the data.
        end_date : Optional[str]
            End date of the data range in the format 'YYYY-MM-DD'. If None, then the data range will end
            at the last date in the data.
        plot : bool, optional
            Whether to plot the persistence diagram.
        distance_method : {'wasserstein', 'bottleneck'}, optional
            The distance method to use for computing persistence diagrams. Default is 'wasserstein'.
        plot : bool, optional
            Whether to plot the computed persistence diagrams. Default is False.

        Returns
        -------
        pd.Series
            The computed moving persistence diagram.

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
                     color='red', label="{0} Distance (scaled)".format(distance_name))

            ax2.set_xlabel('Date')
            ax2.set_ylabel('Topological Distance')
            ax2.set_title('Distance Changes')

            # Adjust spacing between subplots
            plt.tight_layout()

            # Display the plots
            plt.show()

        return distance
