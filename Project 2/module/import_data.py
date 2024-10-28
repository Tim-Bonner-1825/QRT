import pandas as pd

def combined_data_ts_excel(time_series_data):
    """
    Merges the 'Price Data_1' and 'Price Data_2' sheets from an Excel file to create a single
    DataFrame of time series data, handling any overlap.

    Parameters:
    - time_series_data (dict): Dictionary containing data from multiple sheets, with
      'Price Data_1' and 'Price Data_2' as keys for two DataFrames.

    Returns:
    - pandas.DataFrame: A combined DataFrame of time series data, with duplicates removed
      and the 'Date' column standardized to datetime format.
    """
    price_data_1 = time_series_data['Price Data_1']
    price_data_2 = time_series_data['Price Data_2']

    combined_data = pd.concat([price_data_1, price_data_2], ignore_index=True)
    combined_data.drop_duplicates(subset=['Ticker', 'Date'], inplace=True)
    combined_data['Date'] = pd.to_datetime(combined_data['Date'], errors='coerce').astype('datetime64[ms]')

    return combined_data

def return_ticker_symbols():
    """
    Reads the 'Internship_Project_2.xlsx' file to extract and return a list of ticker symbols.

    Returns:
    - list of str: A list containing the ticker symbols from the 'GFD ticker' column.
    """
    df = pd.read_excel(r'Internship_Project_2.xlsx', sheet_name='Sheet1')
    return df['GFD ticker'].tolist()


def create_ts_dictionary(combined_data):
    """
    Creates a dictionary with each ticker symbol as the key and the corresponding
    time series data as the value.

    Parameters:
    - combined_data (pandas.DataFrame): DataFrame containing combined time series data
      with 'Ticker' and 'Date' columns.

    Returns:
    - dict: A dictionary where each key is a ticker symbol, and the value is a DataFrame
      of time series data indexed by date.
    """
    ticker_dict = {}

    for ticker, group in combined_data.groupby('Ticker'):
        time_series = group.set_index('Date')
        ticker_dict[ticker] = time_series

    return ticker_dict


def reorder_ticker_dict(ticker_dict, GFD_tickers):
    """
    Reorders a dictionary of time series data to match the original order of tickers
    in an Excel spreadsheet.

    Parameters:
    - ticker_dict (dict): Dictionary where each key is a ticker symbol and each value is
      a DataFrame of time series data.
    - GFD_tickers (list of str): List of ticker symbols in the desired order.

    Returns:
    - dict: A reordered dictionary of time series data, matching the order of `GFD_tickers`.
    """
    reordered_ticker_dict = {}

    for fx_ticker in GFD_tickers:
        if fx_ticker in ticker_dict:
            reordered_ticker_dict[fx_ticker] = ticker_dict[fx_ticker]

    return reordered_ticker_dict


def average_return_pre_1984(reordered_ticker_dict):
    """
    Calculates the average annual return for each ticker for years prior to 1984.

    Parameters:
    - reordered_ticker_dict (dict): Dictionary where each key is a ticker symbol and each
      value is a DataFrame of time series data, including a 'Close' column.

    Returns:
    - dict: A dictionary where each key is a ticker symbol, and the value is the average
      annual return for the ticker before 1984.
    """
    average_annual_returns_pre_1984 = {}

    for ticker, data in reordered_ticker_dict.items():
        data_before_1984 = data[data.index.year < 1984]
        
        if 'Close' in data_before_1984.columns and not data_before_1984['Close'].isnull().all():
            yearly_data = data_before_1984['Close'].resample('Y').first()
            annual_returns = yearly_data.pct_change().dropna()
            average_return = annual_returns.mean()
            
            average_annual_returns_pre_1984[ticker] = average_return
    
    return average_annual_returns_pre_1984
