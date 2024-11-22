import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PortfolioOptimizer:
    def __init__(self, tickers, start_date=None, end_date=None):
        """
        Initialize the portfolio optimizer with stock tickers and date range
        
        Parameters:
        tickers (list): List of stock ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
    def fetch_data(self):
        """Fetch historical stock data using yfinance"""
        self.data = pd.DataFrame()
        
        for ticker in self.tickers:
            stock = yf.download(ticker, start=self.start_date, end=self.end_date)
            self.data[ticker] = stock['Adj Close']
            
        # Calculate daily returns
        self.returns = self.data.pct_change()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        return self.data
    
    def portfolio_performance(self, weights):
        """
        Calculate portfolio performance metrics
        
        Parameters:
        weights (array): Portfolio weights
        
        Returns:
        tuple: (expected return, volatility, Sharpe ratio)
        """
        returns = np.sum(self.mean_returns * weights) * 252  # Annualized return
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        sharpe_ratio = returns / volatility  # Assuming risk-free rate = 0
        
        return returns, volatility, sharpe_ratio
    
    def negative_sharpe(self, weights):
        """Objective function to minimize (negative Sharpe ratio)"""
        return -self.portfolio_performance(weights)[2]
    
    def optimize_portfolio(self):
        """
        Optimize portfolio weights to maximize Sharpe ratio
        
        Returns:
        dict: Optimized portfolio information
        """
        num_assets = len(self.tickers)
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        )
        bounds = tuple((0, 1) for _ in range(num_assets))  # weights between 0 and 1
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            self.negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Get optimized metrics
        opt_returns, opt_volatility, opt_sharpe = self.portfolio_performance(result.x)
        
        return {
            'weights': dict(zip(self.tickers, result.x)),
            'expected_annual_return': opt_returns,
            'annual_volatility': opt_volatility,
            'sharpe_ratio': opt_sharpe
        }
    
    def plot_efficient_frontier(self, num_portfolios=1000):
        """
        Plot the efficient frontier
        
        Parameters:
        num_portfolios (int): Number of random portfolios to generate
        """
        returns = []
        volatilities = []
        
        for _ in range(num_portfolios):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)
            ret, vol, _ = self.portfolio_performance(weights)
            returns.append(ret)
            volatilities.append(vol)
            
        # Get optimized portfolio
        opt_results = self.optimize_portfolio()
        opt_ret = opt_results['expected_annual_return']
        opt_vol = opt_results['annual_volatility']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(volatilities, returns, c='b', alpha=0.5, label='Random Portfolios')
        plt.scatter(opt_vol, opt_ret, c='r', marker='*', s=200, label='Optimal Portfolio')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.grid(True)
        
        return plt

def main():
    # Example usage
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    optimizer = PortfolioOptimizer(tickers)
    
    # Fetch data
    optimizer.fetch_data()
    
    # Get optimized portfolio
    optimal_portfolio = optimizer.optimize_portfolio()
    
    # Print results
    print("\nOptimal Portfolio Weights:")
    for ticker, weight in optimal_portfolio['weights'].items():
        print(f"{ticker}: {weight:.4f}")
    
    print(f"\nExpected Annual Return: {optimal_portfolio['expected_annual_return']:.4f}")
    print(f"Annual Volatility: {optimal_portfolio['annual_volatility']:.4f}")
    print(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.4f}")
    
    # Plot efficient frontier
    plt = optimizer.plot_efficient_frontier()
    plt.show()

if __name__ == "__main__":
    main()