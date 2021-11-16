# ML-Factor-Model-Simulation

This project was done in collaboration with Andrew Boomer during my masters program at the Toulouse School of Economics.

## Overview
This project was the final project for the Machine Learning class during my masters program. The project involved a replication of the paper, "In-sample inference and forecasting in misspecified factor models" which can be found [here](https://repositori.upf.edu/bitstream/handle/10230/27209/1530.pdf?sequence=1). It focused on in-sample prediction and out-of-sample forecasting in high dimensional models with many exogenous regressors. Four dimension reduction devices are used: Principal Components, Ridge Regression, Landweber Fridman (LF), and Partial Least Squares (PLS). Each involves a regularization or tuning parameter that is selected through generalized cross validation (GCV) or Mallows Cp. Following Carrasco and Rossi we evaluate these estimators in a monte carlo simulation framework with 6 different data generating processes (DGPs). For dimension reduction, we use the singular value decomposition, as well as comparing its computational efficiency in this monte carlo application to the eigenvalue decomposition. These models are then used in an empirical application on a set of monthly macroeconomic and financial indicators.

## Data Sources
The data for the Monte Carlo simulation is based on 6 simulated data generating processes. The data for the empirical application comes from the Federal Reserve (FRED) monthly database.

## Tools
The paper was built using LaTex. The coding was done in python using the Sklearn, Pandas, Seaborn, Statsmodels, and Random packages.
