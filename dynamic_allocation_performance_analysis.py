"""
Python code for blog post "mini-Meucci : Applying The Checklist - Steps 10+"
http://www.returnandrisk.com/2016/07/mini-meucci-applying-checklist-steps-10.html
Copyright (c) 2016  Peter Chan (peter-at-return-and-risk-dot-com)
"""
###############################################################################
# Dynamic Allocation
###############################################################################
#%matplotlib inline
import rnr_meucci_functions as rnr
import numpy as np
from zipline.api import (set_slippage, slippage, set_commission, commission, 
                         order_target_percent, record, schedule_function,
                         date_rules, time_rules, get_datetime, symbol)

# Set tickers for data loading i.e. DJIA constituents and DIA ETF for benchmark
tickers = ['MMM','AXP','AAPL','BA','CAT','CVX','CSCO','KO','DD','XOM','GE','GS',
           'HD','INTC','IBM','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE','PG',
           'TRV','UNH','UTX','VZ','V','WMT','DIS', 'DIA']

# Set investable asset tickers
asset_tickers = ['MMM','AXP','AAPL','BA','CAT','CVX','CSCO','KO','DD','XOM','GE','GS',
        'HD','INTC','IBM','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE','PG',
        'TRV','UNH','UTX','VZ','V','WMT','DIS']
                         
def initialize(context):
    # Turn off the slippage model
    set_slippage(slippage.FixedSlippage(spread=0.0))
    # Set the commission model
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0))
    context.day = -1 # using zero-based counter for days
    context.set_benchmark(symbol('DIA'))
    context.assets = []
    print('Setup investable assets...')
    for ticker in asset_tickers:
        #print(ticker)
        context.assets.append(symbol(ticker))
    context.n_asset = len(context.assets)
    context.n_portfolio = 40 # num mean-variance efficient portfolios to compute
    context.today = None
    context.tau = None
    context.min_data_window = 756 # min of 3 yrs data for calculations
    context.first_rebal_date = None
    context.first_rebal_idx = None
    context.weights = None
    # Schedule dynamic allocation calcs to occur 1 day before month end - note that
    # actual trading will occur on the close on the last trading day of the month
    schedule_function(rebalance,
                  date_rule=date_rules.month_end(days_offset=1),
                  time_rule=time_rules.market_close())
    # Record some stuff every day
    schedule_function(record_vars,
                  date_rule=date_rules.every_day(),
                  time_rule=time_rules.market_close())

def handle_data(context, data):
    context.day += 1
    #print(context.day)
 
def rebalance(context, data):
    # Wait for 756 trading days (3 yrs) of historical prices before trading
    if context.day < context.min_data_window - 1:
        return
    # Get expanding window of past prices and compute returns
    context.today = get_datetime().date() 
    prices = data.history(context.assets, "price", context.day, "1d")
    if context.first_rebal_date is None:
        context.first_rebal_date = context.today
        context.first_rebal_idx = context.day
        print('Starting dynamic allocation simulation...')
    # Get investment horizon in days ie number of trading days next month
    context.tau = rnr.get_num_days_nxt_month(context.today.month, context.today.year)
    # Calculate HFP distribution
    asset_rets = np.array(prices.pct_change(context.tau).iloc[context.tau:, :])
    num_scenarios = len(asset_rets)
    # Set Flexible Probabilities Using Exponential Smoothing
    half_life_prjn = 252 * 2 # in days
    lambda_prjn = np.log(2) / half_life_prjn
    probs_prjn = np.exp(-lambda_prjn * (np.arange(0, num_scenarios)[::-1]))
    probs_prjn = probs_prjn / sum(probs_prjn)
    mu_pc, sigma2_pc = rnr.fp_mean_cov(asset_rets.T, probs_prjn)
    # Perform shrinkage to mitigate estimation risk
    mu_shrk, sigma2_shrk = rnr.simple_shrinkage(mu_pc, sigma2_pc)
    weights, _, _ = rnr.efficient_frontier_qp_rets(context.n_portfolio, 
                                                          sigma2_shrk, mu_shrk)
    print('Optimal weights calculated 1 day before month end on %s (day=%s)' \
        % (context.today, context.day))
    #print(weights)
    min_var_weights = weights[0,:]
    # Rebalance portfolio accordingly
    for stock, weight in zip(prices.columns, min_var_weights):
        order_target_percent(stock, np.asscalar(weight))
    context.weights = min_var_weights
                      
def record_vars(context, data):
    record(weights=context.weights, tau=context.tau)
        
def analyze(perf, bm_value, start_idx):
    pd.DataFrame({'portfolio':results.portfolio_value,'benchmark':bm_value})\
        .iloc[start_idx:,:].plot(title='Portfolio Performance vs Benchmark')

if __name__ == '__main__':
    from datetime import datetime
    import pytz
    from zipline.algorithm import TradingAlgorithm
    from zipline.utils.factory import load_bars_from_yahoo
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Create and run the algorithm.
    algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data)

    start = datetime(2010, 5, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2016, 5, 31, 0, 0, 0, 0, pytz.utc)
    print('Getting Yahoo data for 30 DJIA stocks and DIA ETF as benchmark...')
    data = load_bars_from_yahoo(stocks=tickers, start=start, end=end)
    # Check price data
    data.loc[:, :, 'price'].plot(figsize=(8,7), title='Input Price Data')
    plt.ylabel('price in $');
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.show()

    # Run algorithm
    results = algo.run(data)
    
    # Fix possible issue with timezone
    results.index = results.index.normalize()
    if results.index.tzinfo is None:
        results.index = results.index.tz_localize('UTC')
    
    # Adjust benchmark returns for delayed trading due to 3 year min data window 
    bm_rets = algo.perf_tracker.all_benchmark_returns
    bm_rets[0:algo.first_rebal_idx + 2] = 0
    bm_rets.name = 'DIA'
    bm_rets.index.freq = None
    bm_value = algo.capital_base * np.cumprod(1+bm_rets)
    
    # Plot portfolio and benchmark values
    analyze(results, bm_value, algo.first_rebal_idx + 1)
    print('End value portfolio = {:.0f}'.format(results.portfolio_value.ix[-1]))
    print('End value benchmark = {:.0f}'.format(bm_value[-1]))
    
    # Plot end weights
    pd.DataFrame(results.weights.ix[-1], index=asset_tickers, columns=['w'])\
        .sort_values('w', ascending=False).plot(kind='bar', \
        title='End Simulation Weights', legend=None);

###############################################################################
# Sequel Step - Ex-post performance analysis
###############################################################################
import pyfolio as pf

returns, positions, transactions, gross_lev = pf.utils.\
    extract_rets_pos_txn_from_zipline(results)
trade_start = results.index[algo.first_rebal_idx + 1]
trade_end = datetime(2016, 5, 31, 0, 0, 0, 0, pytz.utc)

print('Annualised volatility of the portfolio = {:.4}'.\
    format(pf.timeseries.annual_volatility(returns[trade_start:trade_end])))
print('Annualised volatility of the benchmark = {:.4}'.\
    format(pf.timeseries.annual_volatility(bm_rets[trade_start:trade_end])))
print('')

pf.create_returns_tear_sheet(returns[trade_start:trade_end], 
                             benchmark_rets=bm_rets[trade_start:trade_end],
                             return_fig=False)
