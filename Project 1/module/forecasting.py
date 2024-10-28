import os
import numpy as np
import pandas as pd
import re
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

def diebold_mariano_test(errors_model, errors_random_walk, horizon=1):
    """
    Calculates the Diebold-Mariano test statistic and p-value to compare forecast accuracy between two models.

    Parameters:
    - errors_model (np.ndarray): Forecast errors from the primary model.
    - errors_random_walk (np.ndarray): Forecast errors from the random walk model.
    - horizon (int): Forecast horizon for the autocovariance adjustment.

    Returns:
    - tuple: A tuple containing the Diebold-Mariano test statistic and its p-value.
    """
    # Calculate the loss differential using squared errors
    loss_diff = (errors_model ** 2) - (errors_random_walk ** 2)
    mean_loss_diff = np.mean(loss_diff)

    # Define autocovariance calculation function
    def autocovariance(x, lag):
        return np.cov(x[:-lag], x[lag:])[0, 1] if lag > 0 else np.var(x)

    T = len(loss_diff)
    gamma_0 = autocovariance(loss_diff, 0)
    sum_gamma = sum([2 * autocovariance(loss_diff, lag) for lag in range(1, horizon)])

    # Calculate DM test statistic and p-value
    dm_stat = mean_loss_diff / np.sqrt((gamma_0 + sum_gamma) / T)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

def generate_forecast(coeffs_df, percentage_changes):
    """
    Generates yearly forecasts for exchange rates based on provided coefficients and yearly changes in variables.

    Parameters:
    - coeffs_df (pd.DataFrame): DataFrame with regression coefficients (intercept and slope) per year.
    - percentage_changes (pd.DataFrame): DataFrame with yearly percentage changes in relevant variables.

    Returns:
    - tuple: A tuple containing:
        - years (list): List of forecast years.
        - forecasted_exchange_rates (np.ndarray): Array of forecasted exchange rates.
    """
    # Identify the year range from the coefficients DataFrame
    min_year = coeffs_df.index.min()
    max_year = coeffs_df.index.max()
    percentage_changes.index = percentage_changes.index.year  # Convert index to years only

    years, forecasted_exchange_rates = [], []

    # Ensure unique entry per year in percentage_changes
    percentage_changes = percentage_changes.groupby(percentage_changes.index).first()

    # Generate forecasts for each year using prior year coefficients
    for year in range(min_year + 1, max_year + 1):
        prior_year = year - 1
        if prior_year in coeffs_df.index and not coeffs_df.loc[prior_year].isna().any():
            intercept = coeffs_df.loc[prior_year, 'Intercept']
            slope = coeffs_df.loc[prior_year, 'Slope']
            if prior_year in percentage_changes.index:
                var_change = percentage_changes.loc[prior_year].values[0]
                if not np.isnan(var_change):
                    forecast = intercept + slope * var_change
                    forecasted_exchange_rates.append(forecast)
                    years.append(year)
    return years, np.array(forecasted_exchange_rates)

def plot_forecasted_exchange_rates(years, forecasted_exchange_rates, ticker, col, ticker_folder):
    """
    Plots and saves forecasted exchange rates over time, performing Diebold-Mariano test on the forecast errors.

    Parameters:
    - years (list): List of forecast years.
    - forecasted_exchange_rates (np.ndarray): Array of forecasted exchange rates.
    - ticker (str): FX ticker identifier.
    - col (str): Name of the economic variable used for forecasting.
    - ticker_folder (str): Path to the directory where the plot should be saved.

    Returns:
    - None
    """
    # Generate random walk forecasts for Diebold-Mariano test
    random_walk_forecasts = np.roll(forecasted_exchange_rates, 1)
    random_walk_forecasts[0] = forecasted_exchange_rates[0]  # Set the first value to match the initial forecast

    # Calculate forecast errors for the model and the random walk
    forecast_errors_model = forecasted_exchange_rates - forecasted_exchange_rates  # Model forecast errors (difference from itself, likely a placeholder)
    forecast_errors_random_walk = forecasted_exchange_rates - random_walk_forecasts  # Random walk forecast errors (difference from shifted values)

    # Perform Diebold-Mariano test to compare the forecast accuracy of the model vs. the random walk
    dm_stat, p_value = diebold_mariano_test(forecast_errors_model, forecast_errors_random_walk)

    # Plot forecasted exchange rates
    plt.figure(figsize=(10, 6))
    plt.plot(years, forecasted_exchange_rates, marker='o', label=f'Forecasted Exchange Rate ({ticker})')
    plt.title(f'Forecasted Exchange Rates ({ticker}) from {col} ({min(years)}-{max(years)})')
    plt.xlabel('Year')
    plt.ylabel('Forecasted Exchange Rate')
    plt.grid(True)
    plt.legend()
    plt.text(0.05, 0.95, f'DM Stat: {dm_stat:.4f}\nP-value: {p_value:.7f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Save plot using sanitized column name
    sanitized_column_name = re.sub(r'[^A-Za-z0-9]+', '_', col)
    plot_file_path = os.path.join(ticker_folder, f'{sanitized_column_name}_forecasted_exchange_rate.jpg')
    plt.savefig(plot_file_path, format='jpg')
    plt.close()
    print(f"Saved forecast plot for {ticker} - {col} at {plot_file_path}")

def process_ticker_data(df_coefficients, df_yearly_pct_changes, folder_name):
    """
    Processes data for each ticker and generates forecasts, storing the results in specified folders.

    Parameters:
    - df_coefficients (pd.DataFrame): DataFrame containing regression coefficients for each ticker and variable.
    - df_yearly_pct_changes (pd.DataFrame): DataFrame containing yearly percentage changes in variables for each ticker.
    - folder_name (str): Path to the main directory where forecast plots are stored.

    Returns:
    - None
    """
    os.makedirs(folder_name, exist_ok=True)

    for ticker in df_coefficients.index:
        if ticker != 'USDUSD':  
            for col in df_coefficients.columns:
                if col not in ['FX_ticker', 'Contract_description', 'Traded_pair']:
                    # Create directory for each ticker
                    ticker_folder = os.path.join(folder_name, ticker)
                    os.makedirs(ticker_folder, exist_ok=True)

                    coeffs_df = df_coefficients[col].loc[ticker]  # Coefficients indexed by year
                    percentage_changes = df_yearly_pct_changes[col].loc[ticker]  # Yearly percentage changes

                    if isinstance(coeffs_df, pd.DataFrame) and not coeffs_df.empty and \
                            isinstance(percentage_changes, pd.DataFrame) and not percentage_changes.empty:
                        
                        # Generate forecasts
                        years, forecasted_exchange_rates = generate_forecast(coeffs_df, percentage_changes)

                        # Plot and save forecasts if sufficient data exists
                        if len(forecasted_exchange_rates) > 1:
                            plot_forecasted_exchange_rates(years, forecasted_exchange_rates, ticker, col, ticker_folder)
                        else:
                            print(f"Not enough data for Random Walk Forecast for {ticker} - {col}. Skipping.")



def calculate_coefficients_over_time(df_yearly_pct_changes, start_year=1984):
    """
    Calculates regression coefficients over time for each ticker/variable pair in the input DataFrame.
    This function iterates over each row in `df_yearly_pct_changes`, performs a regression analysis 
    for each ticker-variable pair over an expanding time window, and stores the results.

    Parameters:
    - df_yearly_pct_changes (pandas.DataFrame): DataFrame containing yearly percentage changes for FX tickers 
      and other variables, where each row is a unique ticker and columns represent variables.

    - start_year (int): The initial year from which to begin the expanding regression analysis. Default is 1984.

    Returns:
    - pandas.DataFrame: DataFrame `df_coefficients` where each cell contains a DataFrame of yearly intercept and 
      slope coefficients for each ticker-variable pair.
    """
    df_coefficients = pd.DataFrame(index=df_yearly_pct_changes.index, columns=df_yearly_pct_changes.columns)

    # Iterate over each row (ticker) in the DataFrame
    for row_index, row in df_yearly_pct_changes.iterrows():
        fx_ticker_df = row['FX_ticker']  # Dependent variable for this row
        
        if isinstance(fx_ticker_df, pd.DataFrame):
            fx_ticker_df.index = pd.to_datetime(fx_ticker_df.index, errors='coerce')
            fx_ticker_df = fx_ticker_df[(fx_ticker_df.index >= '1678-01-01') & (fx_ticker_df.index <= '2262-12-31')]
            fx_ticker_df = fx_ticker_df[fx_ticker_df.index.notna()]  # Remove NaT values

            # Iterate over each independent variable column
            for col in row.index:
                if col not in ['Contract_description', 'FX_ticker']:
                    var_df = row[col]  # Independent variable
                    if isinstance(var_df, pd.DataFrame) and 'Yearly % Change' in var_df.columns:

                        var_df.index = pd.to_datetime(var_df.index, errors='coerce')
                        var_df = var_df[(var_df.index >= '1678-01-01') & (var_df.index <= '2262-12-31')]
                        var_df = var_df[var_df.index.notna()]  # Remove NaT values

                        coefficients_over_time = []

                        # Loop through each year from `start_year` to the max year in the data
                        for year in range(start_year, fx_ticker_df.index.year.max() + 1):
                            y_data = fx_ticker_df['Yearly % Change'][fx_ticker_df.index.year <= year]
                            X_data = var_df['Yearly % Change'][var_df.index.year <= year]

                            combined_data = pd.concat([y_data, X_data], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
                            if combined_data.empty or len(combined_data) <= 1:
                                continue

                            y = combined_data.iloc[:, 0]
                            X = combined_data.iloc[:, 1]
                            X = sm.add_constant(X)

                            # Perform the regression
                            model = sm.OLS(y, X).fit()
                            intercept = model.params.get('const', np.nan)
                            slope = model.params.get('Yearly % Change', np.nan)
                            coefficients_over_time.append([year, intercept, slope])

                        # Create DataFrame to store coefficients by year and save to `df_coefficients`
                        coeffs_df = pd.DataFrame(coefficients_over_time, columns=['Year', 'Intercept', 'Slope']).set_index('Year')
                        df_coefficients.at[row_index, col] = coeffs_df

    return df_coefficients
