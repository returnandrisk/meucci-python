"""
Python code for blog post "mini-Meucci : Applying The Checklist - Steps 6-7"
http://www.returnandrisk.com/2016/06/mini-meucci-applying-checklist-steps-6-7.html
Copyright (c) 2016  Peter Chan (peter-at-return-and-risk-dot-com)
"""
#%matplotlib inline
from pandas_datareader import data
import numpy as np
import pandas as pd
import datetime
import math
import matplotlib.pyplot as plt
import seaborn

# Get Yahoo data on 30 DJIA stocks and a few ETFs
tickers = ['MMM','AXP','AAPL','BA','CAT','CVX','CSCO','KO','DD','XOM','GE','GS',
           'HD','INTC','IBM','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE','PG',
           'TRV','UNH','UTX','VZ','V','WMT','DIS','SPY','DIA','TLT','SHY']
start = datetime.datetime(2008, 4, 1)
end = datetime.datetime(2016, 5, 31)
rawdata = data.DataReader(tickers, 'yahoo', start, end) 
prices = rawdata.to_frame().unstack(level=1)['Adj Close']

###############################################################################
# Quest for Invariance (random walk model) and Estimation (historical approach)
###############################################################################
risk_drivers = np.log(prices)

# Set estimation interval = investment horizon (tau)
tau = 21 # investment horizon in days
invariants = risk_drivers.diff(tau).drop(risk_drivers.index[0:tau])

###############################################################################
# Projection to the Investment Horizon
###############################################################################
# Using the historical simulation approach and setting estimation interval = 
# investment horizon, means that projected invariants = invariants

# Recover the projected scenarios for the risk drivers at the tau-day horizon
risk_drivers_prjn = risk_drivers.loc[end,:] + invariants

###############################################################################
# Pricing at the Investment Horizon
###############################################################################
# Compute the projected $ P&L per unit of each stock for all scenarios
prices_prjn = np.exp(risk_drivers_prjn)
pnl = prices_prjn - prices.loc[end,:]

###############################################################################
# Aggregation at the Investment Horizon
###############################################################################
# Aggregate the individual stock P&Ls into projected portfolio P&L for all scenarios
# Assume equally weighted protfolio at beginning of investment period
capital = 1e6
n_asset = 30
asset_tickers = tickers[0:30]
asset_weights = np.ones(n_asset) / n_asset
# initial holdings ie number of shares
h0 = capital * asset_weights / prices.loc[end, asset_tickers]
pnl_portfolio = np.dot(pnl.loc[:, asset_tickers], h0)

# Apply flexible probabilities to portfolio P&L scenarios
n_scenarios = len(pnl_portfolio)

# Time-conditioned flexible probs with exponential decay
half_life = 252 * 2 # half life of 2 years
es_lambda = math.log(2) / half_life
exp_probs = np.exp(-es_lambda * (np.arange(0, n_scenarios)[::-1]))
exp_probs = exp_probs / sum(exp_probs)
# effective number of scenarios
ens_exp_probs = np.exp(sum(-exp_probs * np.log(exp_probs))) 

# Projected Distribution of Portfolio P&L at Horizon  with flexible probabilities
import rnr_meucci_functions as rnr
mu_port_e, sigma2_port_e = rnr.fp_mean_cov(pnl_portfolio.T, exp_probs)  

###############################################################################
# Ex-ante Evaluation
###############################################################################
# Evaluate the ex-ante portfolio by some satisfaction index
# For example, assume the investor evaluates allocations based only on volatility,
# as measured by standard deviation, and does not take into account expected returns.
# In this case, satisfaction is the opposite of the projected volatility of the portfolio
satisfaction = - np.sqrt(sigma2_port_e)
print('Ex-ante satisfaction index : {:,.0f} (in $ terms)'.format(-np.sqrt(sigma2_port_e)))
print('Ex-ante satisfaction index : {:,.2%} (in % terms)'.format(-np.sqrt(sigma2_port_e)/capital))

###############################################################################
# 7. Ex-ante Attribution
###############################################################################
# Linearly attribute the portfolio ex-ante PnL to the S&P500, long bond ETF + a residual
# Additively attribute the volatility of the portfolio's PnL to S&P500, long bond ETF + a residual

# Set factors 
factor_tickers = ['SPY', 'TLT', 'Residual']
n_factor = 2
# Calculate linear returns
asset_rets = np.array(prices.pct_change(tau).ix[tau:, asset_tickers]) 
factor_rets = np.array(prices.pct_change(tau).ix[tau:, ['SPY', 'TLT']]) 
#port_rets = pnl_portfolio / capital

# Calculate portfolio standard deviation (in percentage terms)
port_std = np.sqrt(sigma2_port_e) / capital

# Factor attribution exposures and risk contributions (using flexible probs)
beta, vol_contr_Z = rnr.factor_attribution(asset_rets, factor_rets, asset_weights, exp_probs, n_factor)

print('Factor exposure (beta):')
for i, factor in enumerate(factor_tickers):
    print('\t\t{}:\t{:.2f}'.format(factor, beta[i]))
print('')
print('Ex-ante portfolio volatility over horizon = {:.2%}'.format(port_std))
print('\tFactor risk contribution:')
for j, factor in enumerate(factor_tickers):
    print('\t\t{}:\t{:.2%}'.format(factor, vol_contr_Z[j]))

# Plot factor risk contribution chart
rnr.plot_waterfall_chart(pd.Series(vol_contr_Z, index=factor_tickers),
                       'Factor Contribution To Portfolio Volatility')






