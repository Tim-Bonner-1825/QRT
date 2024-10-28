import matplotlib.pyplot as plt
import os 
from matplotlib.ticker import MaxNLocator
import pickle 
import pandas as pd
import numpy as np 

def plot_and_save_ticker_data(ticker_dict, valid_tickers, folder_name):
    """
    Generates and saves time series plots for each valid ticker.
    Includes 'Close Price Over Time' and 'Yearly Percentage Change' plots.

    Parameters:
    - ticker_dict (dict): Dictionary containing ticker time series data.
    - valid_tickers (list): List of valid tickers to plot.
    - folder_name (str): Directory name to save the plots.
    """
    os.makedirs(folder_name, exist_ok=True)

    for ticker, ts_data in ticker_dict.items():
        if ticker not in valid_tickers:
            continue  # Skip tickers that are not in the valid list

        # Ensure the time series has 'Date' as the index
        if ts_data.index.name != 'Date':
            ts_data = ts_data.set_index('Date')
        
        # Keep only rows within a reasonable date range and drop NaT values
        ts_data = ts_data[(ts_data.index >= '1678-01-01') & (ts_data.index <= '2262-12-31')]
        ts_data = ts_data[ts_data.index.notna()]

        # Create a folder for each ticker
        ticker_folder = os.path.join(folder_name, ticker)
        os.makedirs(ticker_folder, exist_ok=True)

        # Plot Close Price Over Time
        if 'Close' in ts_data.columns:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
            plt.plot(ts_data.index, ts_data['Close'], label='Close Price')
            plt.title(f'{ticker} Close Price Over Time', pad=20)  # Add padding to the title
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()

            close_price_file_path = os.path.join(ticker_folder, f'{ticker}_time_series.jpg')
            plt.savefig(close_price_file_path, format='jpg')
            plt.close()

            # Plot Yearly Percentage Change with top 5 changes in legend
            plt.figure(figsize=(10, 6))
            yearly_data = ts_data['Close'].resample('YE').ffill()  # Forward fill missing values
            yearly_pct_change = yearly_data.pct_change() * 100  # Convert to percentage

            # Identify the top 5 absolute percentage changes
            abs_pct_changes = yearly_pct_change.abs().nlargest(5)
            top_5_changes = yearly_pct_change.loc[abs_pct_changes.index]

            # Add the top 5 changes to the legend
            top_5_text = "\n".join([f"{year.year}: {change:.2f}%" for year, change in top_5_changes.items()])
            plt.plot(yearly_pct_change.index, yearly_pct_change, marker='o', label=f'{ticker} Yearly % Change\nTop 5 Changes:\n{top_5_text}')
            plt.title(f'{ticker} Yearly Percentage Change Over Time', pad=20)  # Add padding to the title
            plt.xlabel('Year')
            plt.ylabel('Yearly Percentage Change (%)')
            plt.legend(loc='best', fontsize='small', frameon=True)

            pct_change_file_path = os.path.join(ticker_folder, f'{ticker}_yearly_pct_change.jpg')
            plt.savefig(pct_change_file_path, format='jpg')
            plt.close()


def load_and_fill_time_series(ticker_dict_file, excel_file):
    """
    Loads the ticker dictionary from a pickle file and fills in a copy of the provided Excel file
    with time series data based on matching tickers.

    Parameters:
    - ticker_dict_file (str): Path to the pickle file containing the ticker dictionary.
    - excel_file (str): Path to the Excel file to use as a template for filling in time series data.

    Returns:
    - pd.DataFrame: A DataFrame with time series data inserted where tickers match.
    """

    # Load the ticker dictionary from the pickle file
    with open(ticker_dict_file, 'rb') as f:
        ticker_dict = pickle.load(f)

    # Load the Excel file into a DataFrame
    df_project = pd.read_excel(excel_file)
    df_time_series = df_project.copy()

    # Iterate over rows and columns to replace tickers with their time series
    for row_index, row in df_time_series.iterrows():
        for col in df_time_series.columns:
            if col != 'Contract_description':  # Skip the 'Contract_description' column
                ticker = df_time_series.at[row_index, col]  # Get the ticker for this cell

                if pd.notna(ticker) and ticker in ticker_dict:
                    # Replace cell with corresponding DataFrame from ticker_dict
                    df_time_series.at[row_index, col] = ticker_dict[ticker]
                else:
                    # If ticker not found, replace with NaN
                    df_time_series.at[row_index, col] = np.nan


    return df_time_series


def calculate_yearly_pct_changes(df_structure, df_time_series, fx_ticker_column='FX_ticker'):
    """
    Creates a DataFrame with yearly percentage changes for each time series in the given DataFrame.

    Parameters:
    - df_structure (pd.DataFrame): A structure (e.g., from an Excel file) to hold yearly percentage changes.
    - df_time_series (pd.DataFrame): DataFrame with time series data for each ticker.
    - fx_ticker_column (str): Column name for FX tickers to set as index in the result.

    Returns:
    - pd.DataFrame: A DataFrame containing yearly percentage changes for each ticker and variable.
    """
    # Copy the structure to hold the yearly percentage changes
    df_yearly_pct_changes = df_structure.copy()

    # Extract the FX tickers for row indexing
    fx_tickers = df_structure[fx_ticker_column]  # Adjust column name as needed

    # Iterate over rows and columns to calculate yearly percentage changes
    for row_index, row in df_time_series.iterrows():
        fx_ticker_value = fx_tickers[row_index]  # Get the aligned FX ticker value for this row

        for col in df_time_series.columns:
            if col != 'Contract_description':  # Skip the 'Contract_description' column

                ticker_df = df_time_series.at[row_index, col]

                if isinstance(ticker_df, pd.DataFrame) and not ticker_df.empty and 'Close' in ticker_df.columns:
                    # Apply date range filter to ensure valid timestamps
                    ticker_df = ticker_df[(ticker_df.index >= '1678-01-01') & (ticker_df.index <= '2262-12-31')]
                    ticker_df = ticker_df[ticker_df.index.notna()]  # Drop rows with NaT in the index

                    # Resample yearly and calculate percentage change
                    yearly_data = ticker_df['Close'].resample('YS').ffill()  # Forward fill missing values
                    yearly_pct_change = yearly_data.pct_change() * 100  # Convert to percentage

                    # Embed the yearly percentage change DataFrame directly into the cell
                    df_yearly_pct_changes.at[row_index, col] = yearly_pct_change.to_frame(name='Yearly % Change')

    # Set the FX tickers as the index
    df_yearly_pct_changes.index = fx_tickers

    return df_yearly_pct_changes