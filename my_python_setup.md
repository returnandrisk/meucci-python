# My Python Setup
Getting setup with Python for the first time can be a bit painful, so here's what worked for me (as always, your mileage may vary).

## Installation Instructions
OS: Windows 10 (64-bit)
Python Distribution: Anaconda by Continuum https://www.continuum.io/downloads 
	Install Python 3.5 using the Windows 64-bit graphical installer

Once the Anaconda installation is complete, open a command prompt window and type in the following commands to create a Python 3.4 environment (you need to use Python 3.4 for Quantopian's zipline package to work as of May 2016):

conda create --name py34 python=3.4 anaconda
activate py34
conda install -c Quantopian zipline
conda install -c https://conda.anaconda.org/omnia cvxopt
conda install -c quantopian pyfolio 

## Python IDE
Coming from a Matlab/R background, my favourite is spyder.

For writing blog posts and doing presentations, try using a Jupyter Notebook (you can also use it to develop and test code).
To try it out, open a command prompt window and type in:

activate py34
jupyter notebook


