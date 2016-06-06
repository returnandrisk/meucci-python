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
