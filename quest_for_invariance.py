"""
Python code for blog post "mini-Meucci : Applying The Checklist - Step 1"
http://www.returnandrisk.com/2016/06/mini-meucci-applying-checklist-step-1.html
Copyright (c) 2016  Peter Chan (peter-at-return-and-risk-dot-com)
"""
#%matplotlib inline
from pandas_datareader import data
import numpy as np
import datetime
import matplotlib.pyplot as plt
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

# Plots
plt.figure()
prices['AAPL'].plot(figsize=(10, 8), title='AAPL Daily Stock Price (Value)')
plt.show()
plt.figure()
risk_drivers['AAPL'].plot(figsize=(10, 8), 
    title='AAPL Daily Log of Stock Price (Log Value = Risk Driver)')
plt.show()
plt.figure()
invariants['AAPL'].plot(figsize=(10, 8), 
    title='AAPL Continuously Compounded Daily Returns (Log Return = Invariant)')
plt.show()

# Test for invariance using simulated data
import rnr_meucci_functions as rnr
np.random.seed(3)
Data = np.random.randn(1000)
rnr.IIDAnalysis(Data)

# Test for invariance using real data
rnr.IIDAnalysis(invariants.ix[:,'AAPL'])