import numpy as np
import scipy.optimize as sco
import pandas as pd
from pandas import date_range
import matplotlib.pyplot as plt

def portfolio_performance(weights, expected_returns, cov_matrix):
    """
    Returns the expected portfolio return and variance for given weights.

    Parameters:
    - weights (numpy array): Portfolio weights for each asset.
    - expected_returns (numpy array): Expected returns for each asset.
    - cov_matrix (numpy array): Covariance matrix of asset returns.

    Returns:
    - tuple: Expected portfolio return and variance.
    """
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_return, portfolio_variance

def minimize_variance(weights, expected_returns, cov_matrix):
    """
    Objective function to minimize portfolio variance.

    Parameters:
    - weights (numpy array): Portfolio weights for each asset.
    - expected_returns (numpy array): Expected returns for each asset.
    - cov_matrix (numpy array): Covariance matrix of asset returns.

    Returns:
    - float: Portfolio variance.
    """
    return portfolio_performance(weights, expected_returns, cov_matrix)[1]

def constraint_sum_of_weights(weights):
    """
    Constraint to ensure the sum of weights is equal to 1.

    Parameters:
    - weights (numpy array): Portfolio weights for each asset.

    Returns:
    - float: Difference from 1 (should be 0 when constraint is met).
    """
    return np.sum(weights) - 1

def constraint_target_return(weights, expected_returns, target_return, cov_matrix):
    """
    Constraint to ensure portfolio return is equal to the target return.

    Parameters:
    - weights (numpy array): Portfolio weights for each asset.
    - expected_returns (numpy array): Expected returns for each asset.
    - target_return (float): Desired portfolio return.
    - cov_matrix (numpy array): Covariance matrix of asset returns.

    Returns:
    - float: Difference from target return (should be 0 when constraint is met).
    """
    return portfolio_performance(weights, expected_returns, cov_matrix)[0] - target_return

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    """
    Objective function to maximize the Sharpe ratio (minimizing its negative).

    Parameters:
    - weights (numpy array): Portfolio weights for each asset.
    - expected_returns (numpy array): Expected returns for each asset.
    - cov_matrix (numpy array): Covariance matrix of asset returns.
    - risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.

    Returns:
    - float: Negative Sharpe ratio.
    """
    portfolio_return, portfolio_variance = portfolio_performance(weights, expected_returns, cov_matrix)
    portfolio_std_dev = np.sqrt(portfolio_variance)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return -sharpe_ratio

def find_efficient_frontier(target_returns, filtered_expected_returns, filtered_cov_matrix, bounds, initial_guess, original_num_assets, valid_indices):
    """
    Finds efficient frontier by minimizing variance for target returns.

    Parameters:
    - target_returns (numpy array): Range of target returns for efficient frontier.
    - filtered_expected_returns (numpy array): Filtered expected returns for assets.
    - filtered_cov_matrix (numpy array): Filtered covariance matrix for assets.
    - bounds (tuple): Bounds for weights in optimization.
    - initial_guess (numpy array): Initial guess for weights.
    - original_num_assets (int): Total number of assets in the original dataset.
    - valid_indices (numpy array): Indices of assets considered valid (filtered).

    Returns:
    - tuple: Efficient returns, risks, and weights for each target return.
    """
    efficient_risk = []
    efficient_return = []
    efficient_weights = []

    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': constraint_sum_of_weights},
            {'type': 'eq', 'fun': lambda w: portfolio_performance(w, filtered_expected_returns, filtered_cov_matrix)[0] - target_return}
        )

        result = sco.minimize(
            minimize_variance, 
            initial_guess, 
            args=(filtered_expected_returns, filtered_cov_matrix), 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        
        if result.success:
            portfolio_return, portfolio_variance = portfolio_performance(result.x, filtered_expected_returns, filtered_cov_matrix)
            efficient_risk.append(np.sqrt(portfolio_variance))
            efficient_return.append(portfolio_return)

            weights_full = np.zeros(original_num_assets)
            weights_full[valid_indices] = result.x
            efficient_weights.append(weights_full)

    return efficient_return, efficient_risk, efficient_weights

def find_tangent_portfolio(filtered_expected_returns, filtered_cov_matrix, risk_free_rate, initial_guess, bounds, original_num_assets, valid_indices):
    """
    Finds the tangent (maximum Sharpe ratio) portfolio.

    Parameters:
    - filtered_expected_returns (numpy array): Filtered expected returns for assets.
    - filtered_cov_matrix (numpy array): Filtered covariance matrix for assets.
    - risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.
    - initial_guess (numpy array): Initial guess for weights.
    - bounds (tuple): Bounds for weights in optimization.
    - original_num_assets (int): Total number of assets in the original dataset.
    - valid_indices (numpy array): Indices of assets considered valid (filtered).

    Returns:
    - tuple: Tangent portfolio weights, return, risk, and Sharpe ratio.
    """
    tangent_result = sco.minimize(
        negative_sharpe_ratio, 
        initial_guess, 
        args=(filtered_expected_returns, filtered_cov_matrix, risk_free_rate), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=[{'type': 'eq', 'fun': constraint_sum_of_weights}]
    )

    if tangent_result.success:
        tangent_weights_filtered = tangent_result.x
        tangent_return, tangent_variance = portfolio_performance(tangent_weights_filtered, filtered_expected_returns, filtered_cov_matrix)
        tangent_risk = np.sqrt(tangent_variance)
        tangent_sharpe = (tangent_return - risk_free_rate) / tangent_risk

        tangent_weights_full = np.zeros(original_num_assets)
        tangent_weights_full[valid_indices] = tangent_weights_filtered

        return tangent_weights_full, tangent_return, tangent_risk, tangent_sharpe

def generate_covariance_matrix(reordered_ticker_dict, year_chosen):
    """
    Generates covariance and correlation matrices from yearly data post-1984.

    Parameters:
    - reordered_ticker_dict (dict): Dictionary containing time series data for each ticker.
    - year_chosen (int): Year to consider as the end of the data period.

    Returns:
    - tuple: Covariance matrix, correlation matrix, yearly percentage returns, and average annual returns.
    """
    yearly_absolute_change_post_1984 = {}
    yearly_percentage_returns_post_1984 = {}
    average_annual_returns_post_1984 = {}

    for ticker, data in reordered_ticker_dict.items():
        data_after_1984 = data[(data.index.year >= 1984) & (data.index.year <= year_chosen)]
        if not data_after_1984.empty:
            data_after_1984.loc[:, 'Close'] = data_after_1984['Close'].fillna(0)
            yearly_data = data_after_1984['Close'].resample('YS').first()
            # Specify fill_method=None to avoid the FutureWarning
            annual_returns = yearly_data.pct_change(fill_method=None).dropna()
            average_return = annual_returns.mean()
        else:
            years = pd.date_range(start='1984', end=str(year_chosen), freq='YS')
            yearly_data = pd.Series(0, index=years)
            annual_returns = pd.Series(0, index=years[1:])
            average_return = 0

        yearly_absolute_change_post_1984[ticker] = yearly_data
        yearly_percentage_returns_post_1984[ticker] = annual_returns
        average_annual_returns_post_1984[ticker] = average_return

    prices_after_1984 = pd.DataFrame(yearly_percentage_returns_post_1984)
    prices_after_1984.fillna(0, inplace=True)

    cov_matrix = prices_after_1984.cov()
    corr_matrix = prices_after_1984.corr()

    return cov_matrix, corr_matrix, yearly_percentage_returns_post_1984, average_annual_returns_post_1984

def get_next_year_returns(year, over_all_time_percentage_change):
    """
    Retrieves returns for the specified next year from the percentage change data.

    Parameters:
    - year (int): The current year for which to fetch the next year's returns.
    - over_all_time_percentage_change (dict): Dictionary with yearly percentage changes per ticker.

    Returns:
    - numpy array: Array of returns for each ticker in the specified next year.
    """
    next_year_returns = []
    for df in over_all_time_percentage_change.values():
        if (df.index.year == (year + 1)).any():
            next_year_value = df.loc[df.index[df.index.year == (year + 1)][0]]
            next_year_returns.append(next_year_value if isinstance(next_year_value, (int, float)) else 0)
        else:
            next_year_returns.append(0)
    return np.array(next_year_returns)

def compute_optimal_portfolio(expected_returns, cov_matrix, risk_free_rate, initial_guess, bounds, valid_indices):
    """
    Computes the optimal portfolio weights by either finding the tangent portfolio (maximum Sharpe ratio) 
    or, if that fails, selecting the minimum variance portfolio on the efficient frontier.

    Parameters:
    - expected_returns (numpy array): Expected returns for each asset.
    - cov_matrix (pandas DataFrame): Covariance matrix of asset returns.
    - risk_free_rate (float): The risk-free rate to use for Sharpe ratio calculation.
    - initial_guess (numpy array): Initial guess for portfolio weights.
    - bounds (tuple): Bounds for weights in optimization (e.g., no short-selling).
    - valid_indices (numpy array): Boolean array indicating valid assets after filtering.

    Returns:
    - tuple: Contains the portfolio weights, expected return, risk, and Sharpe ratio.
             (weights, expected_return, risk, sharpe_ratio)
             or None if no optimal portfolio could be found.
    """
    # Apply 98th percentile filter to remove extreme returns
    upper_percentile = np.nanpercentile(expected_returns, 98)
    filtered_expected_returns = expected_returns[expected_returns <= upper_percentile]
    filtered_cov_matrix = cov_matrix.loc[valid_indices, valid_indices]

    # Attempt to find the tangent portfolio
    result = find_tangent_portfolio(
        filtered_expected_returns, filtered_cov_matrix, risk_free_rate, initial_guess, bounds, len(expected_returns), valid_indices
    )
    if result is not None:
        return result  # Return tangent portfolio if successful

    # If tangent portfolio fails, find the minimum variance portfolio on the efficient frontier
    print("Using minimum variance portfolio as fallback.")
    target_returns = np.linspace(filtered_expected_returns.min(), filtered_expected_returns.max(), 10)
    efficient_return, efficient_risk, efficient_weights = find_efficient_frontier(
        target_returns, filtered_expected_returns, filtered_cov_matrix, bounds, initial_guess, len(expected_returns), valid_indices
    )
    
    # Find minimum variance portfolio if efficient frontier succeeded
    if efficient_risk:
        min_risk_index = np.argmin(efficient_risk)
        min_variance_weights = efficient_weights[min_risk_index]
        min_variance_return = efficient_return[min_risk_index]
        min_variance_risk = efficient_risk[min_risk_index]
        min_variance_sharpe = (min_variance_return - risk_free_rate) / min_variance_risk
        
        return min_variance_weights, min_variance_return, min_variance_risk, min_variance_sharpe
    else:
        return None


def process_yearly_portfolio_performance(
    start_year=1985,
    end_year=2023,
    risk_free_rate=0.0,
    initial_guess=None,
    bounds=None,
    reordered_ticker_dict=None,
    over_all_time_percentage_change=None,
    capital=1000000  # Initial capital
):
    """
    Simulates yearly portfolio performance, calculating the cumulative capital for each year based on optimal portfolio weights.

    Parameters:
    - start_year (int): The first year of the simulation period (inclusive).
    - end_year (int): The last year of the simulation period (exclusive).
    - risk_free_rate (float): The risk-free rate used for calculating the Sharpe ratio in portfolio optimization.
    - initial_guess (numpy array): Initial guess for portfolio weights in the optimization algorithm.
    - bounds (list of tuples): Constraints for portfolio weights (e.g., no short-selling).
    - reordered_ticker_dict (dict): Dictionary containing reordered ticker data, needed for generating the covariance matrix.
    - over_all_time_percentage_change (dict): Dictionary of percentage changes for each ticker to compute yearly returns.
    - capital (float): Initial capital for the portfolio.

    Returns:
    - tuple: Contains a list of years and the corresponding cumulative portfolio capital for each year.
             (years, cumulative_performance)
    """
    cumulative_performance = [capital]  # Track cumulative performance over the years
    years = list(range(start_year, end_year))  # List of years for plotting

    for year in years:
        print(f"Processing year {year}...")

        # Generate covariance matrix and returns data
        cov_matrix, corr_matrix, yearly_percentage_returns_post_1984, average_annual_returns_post_1984 = generate_covariance_matrix(
            reordered_ticker_dict, year
        )
        expected_returns = np.nan_to_num(np.array(list(average_annual_returns_post_1984.values())), nan=0.0)
        valid_indices = expected_returns <= np.nanpercentile(expected_returns, 98)

        # Compute tangent portfolio or fallback to minimum variance
        tangent_result = compute_optimal_portfolio(expected_returns, cov_matrix, risk_free_rate, initial_guess, bounds, valid_indices)
        if tangent_result is None:
            continue

        tangent_weights, tangent_return, tangent_risk, sharpe_ratio = tangent_result

        # Get next year returns and calculate portfolio return
        next_year_returns = get_next_year_returns(year, over_all_time_percentage_change)
        portfolio_return = float(np.dot(tangent_weights, next_year_returns))

        # Update capital and record cumulative performance
        capital *= (1 + portfolio_return)
        cumulative_performance.append(capital)

        print(f"Year {year + 1} completed. Portfolio return: {portfolio_return:.2%}, Capital: ${capital:,.2f}")

    return years, cumulative_performance

def plot_efficient_frontier_and_tangency(
    efficient_risk,
    efficient_return,
    tangent_risk,
    tangent_return,
    tangent_weights,
    average_annual_returns_post_1984,
    valid_indices,
    sharpe_ratio
):
    """
    Plots the mean-variance efficient frontier with the tangency portfolio and 
    displays tangency portfolio details, including weights, expected return, 
    risk, and Sharpe ratio.

    Parameters:
    - efficient_risk (list or numpy array): Portfolio risks along the efficient frontier.
    - efficient_return (list or numpy array): Portfolio returns along the efficient frontier.
    - tangent_risk (float): Risk (standard deviation) of the tangency portfolio.
    - tangent_return (float): Expected return of the tangency portfolio.
    - tangent_weights (numpy array): Weights of assets in the tangency portfolio.
    - average_annual_returns_post_1984 (dict): Average annual returns of assets.
    - valid_indices (numpy array): Boolean array indicating valid assets for the portfolio.
    - sharpe_ratio (float): Sharpe ratio of the tangency portfolio.

    Returns:
    - None: Displays a plot and prints tangency portfolio details.
    """

    # Plot the Efficient Frontier
    plt.figure(figsize=(10, 6))
    plt.plot(efficient_risk, efficient_return, label='Efficient Frontier', color='b')
    plt.scatter(tangent_risk, tangent_return, marker='*', color='r', s=200, label='Tangency Portfolio')
    plt.plot([0, tangent_risk], [0, tangent_return], 'r--', label='Capital Market Line')
    
    plt.title('Mean-Variance Efficient Frontier with Tangency Portfolio')
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Portfolio Return')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display Tangency Portfolio Weights
    print("\nTangency Portfolio Weights:")
    for i, ticker in enumerate(np.array(list(average_annual_returns_post_1984.keys()))[valid_indices]):
        weight = round(tangent_weights[i], 4)
        if weight != 0:
            print(f"{ticker}: {weight:.4f}")

    # Display Tangency Portfolio Metrics
    print(f"\nTangency Portfolio Expected Return: {tangent_return:.4f}")
    print(f"Tangency Portfolio Risk (Standard Deviation): {tangent_risk:.4f}")
    print(f"Tangency Portfolio Sharpe Ratio: {sharpe_ratio:.4f}")