"""
Python code functions used in blog posts on Attilio Meucci's The Checklist
on www.returnandrisk.com
Copyright (c) 2016  Peter Chan (peter-at-return-and-risk-dot-com)
"""
###############################################################################
# Quest for Invariance
###############################################################################
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

def IIDAnalysis(Data):
    """
    Port of Attilio Meucci's Matlab file IIDAnalysis.m
    https://www.mathworks.com/matlabcentral/fileexchange/25010-exercises-in-advanced-risk-and-portfolio-management
    this function performs simple invariance (i.i.d.) tests on a time series
    1. it checks that the variables are identically distributed by looking at the 
    histogram of two subsamples
    2. it checks that the variables are independent by looking at the 1-lag scatter plot
    under i.i.d. the location-dispersion ellipsoid should be a circle
    see "Risk and Asset Allocation"-Springer (2005), by A. Meucci
    """
    
    # test "identically distributed hypothesis": split observations into two sub-samples and plot histogram
    Sample_1 = Data[0:math.floor(Data.size/2)]
    Sample_2 = Data[math.floor(Data.size/2):]
    num_bins_1 = math.floor(5 * math.log(Sample_1.size))
    num_bins_2 = math.floor(5 * math.log(Sample_2.size))
    X_lim = [Data.min() - .1 * (Data.max() - Data.min()), Data.max() + .1 * (Data.max() - Data.min())]
    n1, xout1 = np.histogram(Sample_1, num_bins_1)
    n2,xout2 = np.histogram(Sample_2, num_bins_2)

    fig=plt.figure(figsize=(9, 8), dpi= 80, facecolor='w', edgecolor='k')
    fig.hold(True)
    gs = gridspec.GridSpec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax.set_position([0.03, .58, .44, .38])
    ax2.set_position([.53, .58, .44, .38])
    ax3.set_position([.31, .08, .38, .38])
    ax.hist(Sample_1, num_bins_1, color=(0.7, 0.7, 0.7), edgecolor='k')
    ax.set_xlim(X_lim)
    ax.set_ylim([0, max([max(n1), max(n2)])])
    ax.set_yticks([])
    ax.set_title(" Distribution 1st Half Sample")
    ax2.hist(Sample_2, num_bins_2, color=(0.7, 0.7, 0.7), edgecolor='k')
    ax2.set_xlim(X_lim)
    ax2.set_ylim([0, max([max(n1), max(n2)])])
    ax2.set_yticks([])
    ax2.set_title("Distribution 2nd Half Sample")

    # test "independently distributed hypothesis": scatter plot of observations at lagged times
    X = Data[0:-1]
    Y = Data[1:]
    ax3.grid(True)
    ax3.scatter(X, Y, s=5, color='#0C63C7', marker='.')
    ax3.set_aspect('equal', 'box')

    tmp = np.column_stack((X, Y))
    m = np.atleast_2d(tmp.mean(0)).transpose()
    S = np.cov(tmp, rowvar=0)
    TwoDimEllipsoid(m,S,2,0,0)
    plt.show()

def TwoDimEllipsoid(Location, Square_Dispersion, Scale, PlotEigVectors, PlotSquare):
    """
    Port of Attilio Meucci's Matlab file TwoDimEllipsoid.m
    https://www.mathworks.com/matlabcentral/fileexchange/25010-exercises-in-advanced-risk-and-portfolio-management
    this function computes the location-dispersion ellipsoid 
    see "Risk and Asset Allocation"-Springer (2005), by A. Meucci
    """

    # compute the ellipsoid in the r plane, solution to  ((R-Location)' * Dispersion^-1 * (R-Location) ) = Scale^2                                   
    EigenValues, EigenVectors = np.linalg.eigh(Square_Dispersion)

    Angle = np.arange(0, 2 * math.pi + math.pi/500, math.pi/500)
    Centered_Ellipse = np.zeros((2, np.size(Angle)), dtype=complex)
    NumSteps = np.size(Angle)
    for i in range(NumSteps):
        # normalized variables (parametric representation of the ellipsoid)
        y = np.array([[math.cos(Angle[i])], [math.sin(Angle[i])]]) 
        Centered_Ellipse[:,i] = (np.dot(np.dot(EigenVectors, np.diag(np.sqrt(EigenValues))), y)).reshape(1,2)

    Centered_Ellipse =  np.real(Centered_Ellipse)
    R = np.dot(Location, np.ones((1, NumSteps))) + Scale * Centered_Ellipse
    plt.plot(R[0,:], R[1,:], color='r', linewidth=2)
    plt.title("Location-Dispersion Ellipsoid")
    plt.xlabel("obs")
    plt.ylabel("lagged obs")

    # plot a rectangle centered in Location with semisides of lengths Dispersion(1) and Dispersion(2), respectively
    if PlotSquare:
        Dispersion = np.sqrt(np.diag(Square_Dispersion))
        Vertex_LowRight_A = Location[0] + Scale * Dispersion[0]
        Vertex_LowRight_B = Location[1] - Scale * Dispersion[1]
        Vertex_LowLeft_A = Location[0] - Scale * Dispersion[0]
        Vertex_LowLeft_B = Location[1] - Scale * Dispersion[1]
        Vertex_UpRight_A = Location[0] + Scale * Dispersion[0]
        Vertex_UpRight_B = Location[1] + Scale * Dispersion[1]
        Vertex_UpLeft_A = Location[0] - Scale * Dispersion[0]
        Vertex_UpLeft_B = Location[1] + Scale * Dispersion[1]
        
        Square = np.array([[Vertex_LowRight_A, Vertex_LowRight_B],
            [Vertex_LowLeft_A, Vertex_LowLeft_B],
            [Vertex_UpLeft_A, Vertex_UpLeft_B],
            [Vertex_UpRight_A, Vertex_UpRight_B],
            [Vertex_LowRight_A, Vertex_LowRight_B]])

        plt.plot(Square[:,0], Square[:,1], color='r', linewidth=2)

    # plot eigenvectors in the r plane (centered in Location) of length the
    # square root of the eigenvalues (rescaled)
    if PlotEigVectors:
        L_1 = Scale * np.sqrt(EigenValues[0])
        L_2 = Scale * np.sqrt(EigenValues[1])
        
        # deal with reflection: matlab chooses the wrong one
        Sign = np.sign(EigenVectors[0,0])
        Start_A = Location[0] # eigenvector 1
        End_A = Location[0] + Sign * (EigenVectors[0,0]) * L_1
        Start_B = Location[1]
        End_B = Location[1] + Sign * (EigenVectors[0,1]) * L_1
        plt.plot([Start_A, End_A], [Start_B, End_B], color='r', linewidth=2)
        
        Start_A = Location[0] # eigenvector 2
        End_A = Location[0] + (EigenVectors[1,0] * L_2)
        Start_B = Location[1]
        End_B = Location[1] + (EigenVectors[1,1] * L_2)
        plt.plot([Start_A, End_A], [Start_B, End_B], color='r', linewidth=2)

    
###############################################################################
# Estimation
###############################################################################
def fp_mean_cov(x, p):
    """
    Computes the HFP-mean and HFP-covariance of the data in x 
    """
    # FP mean
    if x.ndim == 1:
        mu = np.average(x, axis=0, weights=p)
    else:
        mu = np.average(x, axis=1, weights=p)
    # FP covariance
    cov = np.cov(x, aweights=p, ddof=0)
    return((mu, cov))

import seaborn as sns
def plot_corr_heatmap(corr, labels, heading):
    
    sns.set(style="white")
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(8, 8))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
                square=True, xticklabels=labels, yticklabels=labels,
                linewidths=.5, ax=ax, cbar_kws={"shrink": .5}, annot=True)
    ax.set_title(heading)
    plt.show()        

def plot_2_corr_heatmaps(corr1, corr2, labels, title1, title2):
    fig=plt.figure(figsize=(9, 8))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    sns.set(style="white")
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr1, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr1, mask=mask, cmap=cmap, vmax=.3,
                square=True, xticklabels=labels, yticklabels=labels,
                linewidths=.5, ax=ax1, cbar_kws={"shrink": .3}, annot=True)
    ax1.set_title(title1)
    sns.heatmap(corr2, mask=mask, cmap=cmap, vmax=.3,
                square=True, xticklabels=labels, yticklabels=labels,
                linewidths=.5, ax=ax2, cbar_kws={"shrink": .3}, annot=True)
    ax2.set_title(title2)
    fig.tight_layout()
    plt.show()

###############################################################################
# Attribution
###############################################################################
def factor_attribution(asset_rets, factor_rets, asset_weights, probs, N_factors):
    # Ref: http://www.mathworks.com/matlabcentral/fileexchange/26853-factors-on-demand
    # StatisticalVsCrossSectional > S_Main.m
    port_rets = np.dot(asset_rets, asset_weights)
    port_std = np.sqrt(np.cov(port_rets, aweights=probs, ddof=0))
    # Notation: X = asset, Z = factor, P = portfolio, U = residual
    # sigma2 = variance-covariance matrix
    # sigma = covariance terms only
    mu_PZ, sigma2_PZ = fp_mean_cov(np.concatenate((port_rets[:, None], factor_rets), axis=1).T, probs)
    sigma_PZ = sigma2_PZ[0, 1:N_factors+1]
    sigma2_Z = sigma2_PZ[1:N_factors+1, 1:N_factors+1]
    # Compute OLS loadings for the linear return model
    # Compute exposure i.e. beta
    beta = np.dot(np.dot(sigma_PZ.T, sigma2_Z.T), np.linalg.inv(np.dot(sigma2_Z, sigma2_Z.T)))
    mu_P = mu_PZ[0]
    mu_Z = mu_PZ[1:N_factors+1]
    alpha = mu_P - np.dot(beta, mu_Z)
    # Compute residuals
    U = port_rets - alpha - np.dot(factor_rets, beta)
    # Compute risk contribution
    mu_ZU, sigma2_ZU = fp_mean_cov(np.concatenate((factor_rets, U[:, None]), axis=1).T, probs)
    beta_ = np.append(beta, 1)
    vol_contr_Z = beta_ * np.dot(sigma2_ZU, beta_) / port_std
    return(beta_, vol_contr_Z)
    
def plot_waterfall_chart(series, title):
    df = pd.DataFrame({'pos':np.maximum(series,0),'neg':np.minimum(series,0)})
    blank = series.cumsum().shift(1).fillna(0)
    df.plot(kind='bar', stacked=True, bottom=blank, title=title, figsize=(9, 8)) 

###############################################################################
# Construction
###############################################################################
def simple_shrinkage(mu, cov, mu_shrk_wt=0.1, cov_shrk_wt=0.1):
    # Reference: Attilio Meucci's Matlab file S_MVHorizon.m
    # https://www.mathworks.com/matlabcentral/fileexchange/25010-exercises-in-advanced-risk-and-portfolio-management
    n_asset = len(mu)
    
    # Mean shrinkage 
    Shrk_Exp = np.zeros(n_asset)
    Exp_C_Hat = (1 - mu_shrk_wt) * mu + mu_shrk_wt * Shrk_Exp
    
    # Covariance shrinkage
    Shrk_Cov = np.eye(n_asset) * np.trace(cov) / n_asset
    Cov_C_Hat = (1-cov_shrk_wt) * cov + cov_shrk_wt * Shrk_Cov
    
    return((Exp_C_Hat, Cov_C_Hat))

def efficient_frontier_qp_rets(n_portfolio, covariance, expected_values):
    """
    Port of Attilio Meucci's Matlab file EfficientFrontierQPRets.m
    https://www.mathworks.com/matlabcentral/fileexchange/25010-exercises-in-advanced-risk-and-portfolio-management
    This function returns the n_portfolio x 1 vector expected returns,
                          the n_portfolio x 1 vector of volatilities and
                          the n_portfolio x n_asset matrix of weights
    of n_portfolio efficient portfolios whose expected returns are equally spaced along the whole range of the efficient frontier
    """
    import cvxopt as opt
    from cvxopt import solvers, blas
    
    solvers.options['show_progress'] = False
    n_asset = covariance.shape[0]
    expected_values = opt.matrix(expected_values)
    
    # determine weights, return and volatility of minimum-risk portfolio
    S = opt.matrix(covariance)
    pbar = opt.matrix(np.zeros(n_asset)) 
    # 1. positive weights
    G = opt.matrix(0.0, (n_asset, n_asset))
    G[::n_asset+1] = -1.0
    h = opt.matrix(0.0, (n_asset, 1))
    # 2. weights sum to one
    A = opt.matrix(1.0, (1, n_asset))
    b = opt.matrix(1.0)
    x0 = opt.matrix(1 / n_asset * np.ones(n_asset))
    min_x = solvers.qp(S, pbar, G, h, A, b, 'coneqp', x0)['x']
    min_ret = blas.dot(min_x.T, expected_values)
    min_vol = np.sqrt(blas.dot(min_x, S * min_x))
    
    # determine weights, return and volatility of maximum-risk portfolio
    max_idx = np.asscalar(np.argmax(expected_values))
    max_x = np.zeros(n_asset)
    max_x[max_idx] = 1
    max_ret = expected_values[max_idx]
    max_vol = np.sqrt(np.dot(max_x, np.dot(covariance, max_x)))
    
    # slice efficient frontier returns into n_portfolio segments
    target_rets = np.linspace(min_ret, max_ret, n_portfolio).tolist()
    
    # compute the n_portfolio weights and risk-return coordinates of the optimal allocations for each slice
    weights = np.zeros((n_portfolio, n_asset))
    rets = np.zeros(n_portfolio)
    vols = np.zeros(n_portfolio)
    # start with min vol portfolio
    weights[0,:] = np.asarray(min_x).T
    rets[0] = min_ret
    vols[0] = min_vol
    
    for i in range(1, n_portfolio-1):
        # determine least risky portfolio for given expected return
        A = opt.matrix(np.vstack([np.ones(n_asset), expected_values.T]))
        b = opt.matrix(np.hstack([1, target_rets[i]]))
        x = solvers.qp(S, pbar, G, h, A, b, 'coneqp', x0)['x']    
        weights[i,:] = np.asarray(x).T
        rets[i] = blas.dot(x.T, expected_values)
        vols[i] = np.sqrt(blas.dot(x, S * x))    
    
    # add max ret portfolio
    weights[n_portfolio-1,:] = np.asarray(max_x).T
    rets[n_portfolio-1] = max_ret
    vols[n_portfolio-1] = max_vol
    
    return(weights, rets, vols)

###############################################################################
# Dynamic Allocation
###############################################################################
from zipline.utils.tradingcalendar import get_trading_days
from datetime import datetime
import pytz

def get_num_days_nxt_month(month, year):
    """
    Inputs: today's month number and year number
    Output: number of trading days in the following month
    """
    nxt_month = month + 1 if month < 12 else 1
    _year = year if nxt_month != 1 else year + 1
    start = datetime(_year, nxt_month, 1, tzinfo=pytz.utc)
    end = datetime(_year if nxt_month != 12 else _year + 1, nxt_month + 1 if nxt_month != 12 else 1, 1, tzinfo=pytz.utc)
    return(len(get_trading_days(start, end)))
    

