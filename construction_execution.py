"""
Python code for blog post "mini-Meucci : Applying The Checklist - Steps 8-9"
http://www.returnandrisk.com/2016/06/mini-meucci-applying-checklist-steps-8-9.html
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

# Setup
tau = 21 # investment horizon in days
n_scenarios = len(prices) - tau
n_asset = 30
asset_tickers = tickers[0:30]

###############################################################################
# Construction - 2 step mean-variance optimization
###############################################################################
# Take shortcut and bypass some of the checklist steps in this toy example since
# returns are invariants, estimation interval = horizon ie can use linear return
# distribution directly as input into mean-variance optimizer

# Projected linear returns to the horizon - historical simulation 
asset_rets = np.array(prices.pct_change(tau).ix[tau:, asset_tickers]) 

# Mean-variance inputs
# Distribution of asset returns at horizon with flexible probabilities
# Time-conditioned flexible probs with exponential decay
half_life = 252 * 2 # half life of 2 years
es_lambda = math.log(2) / half_life
exp_probs = np.exp(-es_lambda * (np.arange(0, n_scenarios)[::-1]))
exp_probs = exp_probs / sum(exp_probs)

# Apply flexible probabilities to asset return scenarios
import rnr_meucci_functions as rnr
mu_pc, sigma2_pc = rnr.fp_mean_cov(asset_rets.T, exp_probs)

# Perform shrinkage to mitigate estimation risk
mu_shrk, cov_shrk = rnr.simple_shrinkage(mu_pc, sigma2_pc)

# Step 1: m-v quadratic optimization for efficient frontier
n_portfolio = 40
weights_pc, rets_pc, vols_pc = rnr.efficient_frontier_qp_rets(n_portfolio, 
                                                              cov_shrk, mu_shrk)

# Step 2: evaluate satisfaction for all allocations on the frontier
satisfaction_pc = -vols_pc

# Choose the allocation that maximises satisfaction
max_sat_idx = np.asscalar(np.argmax(satisfaction_pc))
max_sat = satisfaction_pc[max_sat_idx]
max_sat_weights = weights_pc[max_sat_idx, :]
print('Optimal portfolio is minimum volatility portfolio with satisfaction\
 index = {:.2}'.format(max_sat))

# Plot charts
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(9, 8))
fig.hold(True)
gs = gridspec.GridSpec(2, 1)
ax = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax.plot(vols_pc, rets_pc)
ax.set_xlim(vols_pc[0]*0.95, vols_pc[-1]*1.02)
ax.set_ylim(min(rets_pc)*0.9, max(rets_pc)*1.05)
ax.set_xlabel('Standard Deviation')
ax.set_ylabel('Expected Return')
ax.set_title("Efficient Frontier")
ax.plot(vols_pc[0], rets_pc[0], 'g.', markersize=10.0)
ax.text(vols_pc[0]*1.02, rets_pc[0], 'minimum volatility portfolio',
         fontsize=10)

ax2.plot(vols_pc, satisfaction_pc)
ax2.set_xlim(vols_pc[0]*0.95, vols_pc[-1]*1.02)
ax2.set_ylim(min(satisfaction_pc)*1.05, max(satisfaction_pc)*0.9)
ax2.set_xlabel('Standard Deviation')
ax2.set_ylabel('Satisfaction')
ax2.set_title("Satisfaction")
ax2.plot(vols_pc[max_sat_idx], max(satisfaction_pc), 'g.', markersize=10.0)
ax2.text(vols_pc[max_sat_idx]*1.02, max(satisfaction_pc), 'maximum satisfaction',
         fontsize=10)
plt.tight_layout()
plt.show()    

# Plot minimum volatility portfolio weights
pd.DataFrame(weights_pc[0,:], index=asset_tickers, columns=['w']).sort_values('w', \
    ascending=False).plot(kind='bar', title='Minimum Volatility Portfolio Weights', \
    legend=None, figsize=(10, 8))
plt.show()

###############################################################################
# Execution
###############################################################################
# See zipline simulation in dynamic allocation code file