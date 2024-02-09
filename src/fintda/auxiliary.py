# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
Auxiliary functions for the fintda package
==========================================
*Created on 25.12.2023 by bari_is*
*Copyright (C) 2023*
*For COPYING and LICENSE details, please refer to the LICENSE file*

This module contains auxiliary functions for the fintda package.
"""
import os
from typing import Optional

import pandas as pd

__all__ = ['get_data_range']


__PATH__ = os.path.join(os.path.dirname(__file__), "data", "data.csv")


def get_data_range(data: pd.DataFrame,
                   begin_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   days: Optional[int] = None) -> pd.DataFrame:
    """
    Get a subset of data based on the specified date range or number of days.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    begin_date : str, optional
        The start date of the data range in the format 'YYYY-MM-DD'.
    end_date : str, optional
        The end date of the data range in the format 'YYYY-MM-DD'.
    days : int, optional
        The number of days to consider for the data range (see notes).

    Returns
    -------
    pd.DataFrame
        The subset of data based on the specified date range or number of days.

    Notes
    -----
    If `days` and `begin_date` is specified, then the data range will be from `begin_date` to  
    `begin_date + days`.
    """
    if begin_date is None and days is not None:
        begin_date = data.index.min()
        end_date = begin_date + pd.DateOffset(days=days)
        return data.loc[begin_date:end_date]

    if begin_date is not None and days is not None:
        begin_date = pd.to_datetime(begin_date)
        end_date = begin_date + pd.DateOffset(days=days)
        return data.loc[begin_date:end_date]

    if begin_date is not None and end_date is not None:
        return data.loc[begin_date:end_date]

    if begin_date is not None:
        return data.loc[begin_date:]

    if end_date is not None:
        return data.loc[:end_date]

    return data


def load_data() -> pd.DataFrame:
    """
    A auxiliary function to load saved test data.

    Returns
    -------
    out : DataFrame
    """
    data = pd.read_csv(__PATH__)
    data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')
    data = data.set_index("Date")

    return data
