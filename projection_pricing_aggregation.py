"""
Python code for blog post "mini-Meucci : Applying The Checklist - Steps 3-5"
http://www.returnandrisk.com/2016/06/mini-meucci-applying-checklist-steps-3-5.html
Copyright (c) 2016  Peter Chan (peter-at-return-and-risk-dot-com)
"""
#%matplotlib inline
from pandas_datareader import data
import numpy as np
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

# Equal probs
equal_probs = np.ones(n_scenarios) / n_scenarios

# Time-conditioned flexible probs with exponential decay
half_life = 252 * 2 # half life of 2 years
es_lambda = math.log(2) / half_life
exp_probs = np.exp(-es_lambda * (np.arange(0, n_scenarios)[::-1]))
exp_probs = exp_probs / sum(exp_probs)
# effective number of scenarios
ens_exp_probs = np.exp(sum(-exp_probs * np.log(exp_probs))) 

# Projected Distribution of Portfolio P&L at Horizon  with flexible probabilities
import rnr_meucci_functions as rnr
mu_port, sigma2_port = rnr.fp_mean_cov(pnl_portfolio.T, equal_probs)  
mu_port_e, sigma2_port_e = rnr.fp_mean_cov(pnl_portfolio.T, exp_probs)  

print('Ex-ante portfolio $P&L mean over horizon (equal probs) : {:,.0f}'.format(mu_port))
print('Ex-ante portfolio $P&L volatility over horizon (equal probs) : {:,.0f}'.format(np.sqrt(sigma2_port)))
print('')
print('Ex-ante portfolio $P&L mean over horizon (flex probs) : {:,.0f}'.format(mu_port_e))
print('Ex-ante portfolio $P&L volatility over horizon (flex probs) : {:,.0f}'.format(np.sqrt(sigma2_port_e)))

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111)
ax.hist(pnl_portfolio, 50, weights=exp_probs) 
ax.set_title('Ex-ante Distribution of Portfolio P&L (flexbile probabilities with exponential decay)') 
plt.show()






