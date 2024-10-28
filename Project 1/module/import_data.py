import pandas as pd 
import os 
import re 

def return_excel_tickers():
    """
    Reads the 'Internship_Project_1.xlsx' file to retrieve a list of FX tickers. 
    Additionally, saves each column of the Excel file as an individual CSV file 
    within the './gfd_input_csvs' directory.

    Returns:
    - list: A list of FX tickers if the 'FX_ticker' column exists in the Excel file.
    """
    df_project_1 = pd.read_excel(r'Internship_Project_1.xlsx')
    output_dir = "./gfd_input_csvs"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save each column in the Excel file as an individual CSV file without header or index
    for column in df_project_1.columns:
        sanitized_column_name = re.sub(r'[^A-Za-z0-9]+', '_', column)
        output_file_path = os.path.join(output_dir, f"{sanitized_column_name}.csv")
        df_project_1[[column]].to_csv(output_file_path, index=False, header=False)
        print(f"Saved column '{column}' to {output_file_path}")  # Debug message

    # Check if 'FX_ticker' column exists and return it as a list if so
    if 'FX_ticker' in df_project_1.columns:
        tickers_list = df_project_1['FX_ticker'].dropna().tolist()  # Drop NaNs to ensure clean list
        print("FX Tickers:", tickers_list)  # Debug message
        return tickers_list
    else:
        print("FX_ticker column not found in the Excel file.")
        return []


def filter_and_save_csv(file_path, filter_value):
    """
    Reads a CSV file, filters out rows where the first column matches the specified filter value, 
    and saves the filtered result back to the same file without headers or indexes.

    Parameters:
    - file_path (str): Path to the CSV file to be filtered.
    - filter_value (str): The value in the first column to exclude from the file.
    """
    df = pd.read_csv(file_path, header=None)  # Read without header to treat all rows as data
    df.columns = ['Column1']  # Assuming only one column; adjust if there are more

    # Filter out rows where the first column matches the filter_value
    df_filtered = df[df.iloc[:, 0] != filter_value]
    
    # Save back to CSV without header or index
    df_filtered.to_csv(file_path, index=False, header=False)


def process_excel_sheets(excel_sheets_folder, df_project, unfound_tickers):
    """
    Loads time series data for each column in the provided DataFrame by reading
    corresponding Excel sheets from the specified folder. Combines available sheets
    and processes data for each ticker. If a ticker is found in 'unfound_tickers',
    the data is inverted.

    Parameters:
    - excel_sheets_folder (str): Directory containing the Excel files.
    - df_project (pd.DataFrame): DataFrame with ticker names as columns.
    - unfound_tickers (list): List of tickers requiring inversion.

    Returns:
    - dict: Dictionary with tickers as keys and processed time series DataFrames as values.
    """
    ticker_dict = {}

    for column_name in df_project.columns:
        sanitized_column_name = re.sub(r'[^A-Za-z0-9]+', '_', column_name)
        file_path = os.path.join(excel_sheets_folder, f'{sanitized_column_name}.xlsx')

        if os.path.exists(file_path):
            time_series_data = pd.read_excel(file_path, sheet_name=None)
            
            # Combine data from available sheets
            if {'Price Data_1', 'Price Data_2', 'Price Data_3'}.issubset(time_series_data):
                combined_data = concatenate_and_clean_data(
                    time_series_data['Price Data_1'],
                    time_series_data['Price Data_2'],
                    time_series_data['Price Data_3']
                )
            elif {'Price Data_1', 'Price Data_2'}.issubset(time_series_data):
                combined_data = concatenate_and_clean_data(
                    time_series_data['Price Data_1'],
                    time_series_data['Price Data_2']
                )
            elif 'Price Data' in time_series_data:
                combined_data = time_series_data['Price Data']
                combined_data['Date'] = pd.to_datetime(combined_data['Date'], errors='coerce')
                combined_data.drop_duplicates(subset=['Ticker', 'Date'], inplace=True)
            else:
                continue
            
            # Group data by ticker and handle reversal/inversion if necessary
            for ticker, group in combined_data.groupby('Ticker'):  
                time_series = group.set_index('Date')
                reversed_ticker = ticker[3:] + ticker[:3]

                if reversed_ticker in unfound_tickers:
                    time_series = invert_time_series(time_series)
                    ticker_dict[reversed_ticker] = time_series
                else:
                    ticker_dict[ticker] = time_series

    return ticker_dict

def extend_main_with_inverted(excel_sheets_folder, ticker_dict):
    """
    Extends the time series in the main ticker dictionary with data from the
    'inverted_FX_tickers' file if the inverted series has earlier data.

    Parameters:
    - excel_sheets_folder (str): Directory containing the Excel files.
    - ticker_dict (dict): Dictionary containing main tickers with time series data.

    Returns:
    - dict: Updated ticker_dict with extended time series where applicable.
    """
    # Path to the inverted FX tickers file
    inverted_file_path = os.path.join(excel_sheets_folder, 'inverted_FX_tickers.xlsx')

    # Load and process the inverted FX tickers if the file exists
    if os.path.exists(inverted_file_path):
        inverted_data = pd.read_excel(inverted_file_path, sheet_name=None)

        # Combine 'Price Data' sheets if they exist
        if {'Price Data_1', 'Price Data_2', 'Price Data_3'}.issubset(inverted_data):
            combined_inverted_data = pd.concat([
                inverted_data['Price Data_1'],
                inverted_data['Price Data_2'],
                inverted_data['Price Data_3']
            ])
        elif {'Price Data_1', 'Price Data_2'}.issubset(inverted_data):
            combined_inverted_data = pd.concat([
                inverted_data['Price Data_1'],
                inverted_data['Price Data_2']
            ])
        elif 'Price Data' in inverted_data:
            combined_inverted_data = inverted_data['Price Data']
        else:
            return ticker_dict  # No relevant sheets found, return unmodified ticker_dict

        # Clean combined data
        combined_inverted_data['Date'] = pd.to_datetime(combined_inverted_data['Date'], errors='coerce')
        combined_inverted_data.drop_duplicates(subset=['Ticker', 'Date'], inplace=True)
        combined_inverted_data.set_index('Date', inplace=True)

        # Process each ticker in the combined inverted data
        for ticker, group in combined_inverted_data.groupby('Ticker'):
            # Determine the main ticker format (e.g., AUDUSD instead of USDAUD)
            main_ticker = ticker[3:] + ticker[:3]

            # Check if the main ticker exists in ticker_dict
            if main_ticker in ticker_dict:
                # Get the main time series and the inverted time series
                main_series = ticker_dict[main_ticker]
                inverted_series = group.copy()

                # Invert the price data in the inverted series
                inverted_series['Price'] = 1 / inverted_series['Price']

                # Check if the inverted series has earlier data
                if inverted_series.index.min() < main_series.index.min():
                    print('Inverted')
                    # Concatenate inverted data before main data to extend it
                    extended_series = pd.concat([inverted_series, main_series]).sort_index().drop_duplicates()

                    # Update ticker_dict with the extended series
                    ticker_dict[main_ticker] = extended_series

    return ticker_dict


def concatenate_and_clean_data(*dfs):
    """
    Concatenates multiple DataFrames, ensures 'Date' column is in datetime format, 
    and removes duplicate rows based on 'Ticker' and 'Date'.

    Parameters:
    - *dfs (pd.DataFrame): Variable number of DataFrames to concatenate and clean.

    Returns:
    - pd.DataFrame: A single combined and cleaned DataFrame.
    """
    for df in dfs:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    combined_data = pd.concat(dfs, ignore_index=True)
    combined_data.drop_duplicates(subset=['Ticker', 'Date'], inplace=True)
    return combined_data


def invert_time_series(time_series):
    """
    Inverts the open, close, low, and high prices of a time series DataFrame.

    Parameters:
    - time_series (pd.DataFrame): DataFrame with price columns to be inverted.

    Returns:
    - pd.DataFrame: DataFrame with inverted open, close, low, and high prices.
    """
    time_series['Open'] = 1 / time_series['Open']
    time_series['Close'] = 1 / time_series['Close']
    time_series['Low'] = 1 / time_series['Low']
    time_series['High'] = 1 / time_series['High']
    return time_series