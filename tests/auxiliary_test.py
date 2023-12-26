import pandas as pd
import pytest

from financetda.auxiliary import get_data_range, load_data

# Load test data
data = load_data()


def test_get_data_range_with_begin_date():
    begin_date = '2020-01-01'
    expected_data = data.loc[begin_date:]
    assert get_data_range(data, begin_date=begin_date).equals(expected_data)


def test_get_data_range_with_end_date():
    end_date = '2023-01-31'
    expected_data = data.loc[:end_date]
    assert get_data_range(data, end_date=end_date).equals(expected_data)


def test_get_data_range_with_begin_and_end_date():
    begin_date = '2023-01-01'
    end_date = '2023-01-31'
    expected_data = data.loc[begin_date:end_date]
    assert get_data_range(data, begin_date=begin_date, end_date=end_date).equals(expected_data)


def test_get_data_range_with_days():
    days = 30
    begin_date = data.index.min()
    end_date = begin_date + pd.DateOffset(days=days)
    expected_data = data.loc[begin_date:end_date]
    assert get_data_range(data, days=days).equals(expected_data)


def test_get_data_range_with_begin_date_and_days():
    begin_date = '2023-01-01'
    days = 30
    begin_date = pd.to_datetime(begin_date)
    end_date = begin_date + pd.DateOffset(days=days)
    expected_data = data.loc[begin_date:end_date]
    assert get_data_range(data, begin_date=begin_date, days=days).equals(expected_data)


def test_get_data_range_with_no_parameters():
    expected_data = data
    assert get_data_range(data).equals(expected_data)
