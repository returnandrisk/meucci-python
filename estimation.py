"""
Python code for blog post "mini-Meucci : Applying The Checklist - Step 2"
http://www.returnandrisk.com/2016/06/mini-meucci-applying-checklist-step-2.html
Copyright (c) 2016  Peter Chan (peter-at-return-and-risk-dot-com)
"""
#%matplotlib inline
from pandas_datareader import data
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn

# Get Yahoo data on 30 DJIA stocks and a few ETFs
tickers = ['MMM','AXP','AAPL','BA','CAT','CVX','CSCO','KO','DD','XOM','GE','GS',
           'HD','INTC','IBM','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE','PG',
           'TRV','UNH','UTX','VZ','V','WMT','DIS','SPY','DIA','TLT','SHY']
start = datetime.datetime(2005, 12, 31)
end = datetime.datetime(2016, 5, 30)
rawdata = data.DataReader(tickers, 'yahoo', start, end) 
prices = rawdata.to_frame().unstack(level=1)['Adj Close']
risk_drivers = np.log(prices)
invariants = risk_drivers.diff().drop(risk_drivers.index[0])
T = len(invariants)

# Get VIX data
vix = data.DataReader('^VIX', 'yahoo', start, end)['Close']
vix.drop(vix.index[0], inplace=True)

# Equal probs
equal_probs = np.ones(len(vix)) / len(vix)

# Time-conditioned flexible probs with exponential decay
half_life = 252 * 2 # half life of 2 years
es_lambda = math.log(2) / half_life
exp_probs = np.exp(-es_lambda * (np.arange(0, len(vix))[::-1]))
exp_probs = exp_probs / sum(exp_probs)
# effective number of scenarios
ens_exp_probs = math.exp(sum(-exp_probs * np.log(exp_probs))) 

# State-conditioned flexible probs based on VIX > 20
state_probs = np.zeros(len(vix)) / len(vix)
state_cond = np.array(vix > 20)
state_probs[state_cond] = 1 / state_cond.sum()

# Plot charts
fig = plt.figure(figsize=(9, 8))
gs = gridspec.GridSpec(2, 2)
ax = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax.plot(vix.index, equal_probs)
ax.set_title("Equal Probabilities (weights)")
ax2.plot(vix.index, vix)
ax2.set_title("Implied Volatility Index (VIX)")
ax2.axhline(20, color='r')
ax3.plot(vix.index, exp_probs)
ax3.set_title("Time-conditioned Probabilities with Exponential Decay")
ax4.plot(vix.index, state_probs, marker='o', markersize=3, linestyle='None', alpha=0.7)
ax4.set_title("State-conditioned Probabilities (VIX > 20)")
plt.tight_layout()
plt.show()

# Stress analysis
import rnr_meucci_functions as rnr

tmp_tickers = ['AAPL', 'JPM', 'WMT', 'SPY', 'TLT']

# HFP distribution of invariants using equal probs
mu, sigma2 = rnr.fp_mean_cov(invariants.ix[:,tmp_tickers].T, equal_probs)

# HFP distribution of invariants using state-conditioned probs (VIX > 20)
mu_s, sigma2_s = rnr.fp_mean_cov(invariants.ix[:,tmp_tickers].T, state_probs)

# Calculate correlations
from statsmodels.stats.moment_helpers import cov2corr
corr = cov2corr(sigma2)
corr_s = cov2corr(sigma2_s)

# Plot correlation heatmaps
rnr.plot_2_corr_heatmaps(corr, corr_s, tmp_tickers, 
                     "HFP Correlation Heatmap - equal probs",
                     "HFP Correlation Heatmap - state probs (VIX > 20)")