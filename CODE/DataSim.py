import random
import numpy as np
import re
import statsmodels.api as sm

sim_params = {'N' : 200,
	'T' : 500,
	'r' : [4, 50, 5, 5, 200, 1],
	'r_max' : [14, 200, 15, 15, 200, 11],
	'r_iter' : 1}

def gen_params(sim_params):

	N = sim_params['N']
	T = sim_params['T']
	r = sim_params['r']
	r_max = sim_params['r_max']
	r_iter = sim_params['r_iter']

	M = np.divide(np.ones((N, N)), np.array(range(1, (N + 1))).reshape((N, 1)))

	mean = np.zeros(N)
	covar = np.multiply(np.array([1, 2, 3, 3, 4] + [1] * (N - 5)).reshape((1, N)), np.identity(N))

	params = {'DGP1' : {'theta' : np.ones((r[0], 1)), 'nu' : np.random.normal(0, 1, (T, 1)), 'xi' : np.random.normal(0, 1, (T, N)), 'F' : np.random.normal(0, 1, (T, r[0])), 'lambda' : np.random.normal(0, 1, (r[0], N))},
	'DGP2' : {'theta' : np.ones((r[1], 1)), 'nu' : np.random.normal(0, 1, (T, 1)), 'xi' : np.random.normal(0, 1, (T, N)), 'F' : np.random.normal(0, 1, (T, r[1])), 'lambda' : np.random.normal(0, 1, (r[1], N))},
	'DGP3' : {'theta' : np.array([1] + [0] * (r[2] - 1)).reshape((r[2], 1)), 'nu' : np.random.normal(0, 0.01, (T, 1)), 'xi' : np.random.normal(0, 1, (T, N)), 'F' : np.random.multivariate_normal(mean, covar, 10), 'lambda' : np.random.normal(0, 1, (r[2], N))},
	'DGP4' : {'theta' : np.zeros((r[3], 1)), 'nu' : np.random.normal(0, 1, (T, 1)), 'xi' : np.random.normal(0, 1, (T, N)), 'F' : np.random.multivariate_normal(mean, covar, 10), 'lambda' : np.random.normal(0, 1, (r[3], N))},
	'DGP5' : {'theta' : np.zeros((r[4], 1)), 'nu' : np.random.normal(0, 1, (T, 1)), 'xi' : np.random.normal(0, 1, (T, N)), 'F' : np.random.normal(0, 1, (T, r[4])), 'lambda' : np.multiply(M, np.random.normal(0, 1, (N, N)))},
	'DGP6' : {'theta' : np.ones((r[5], 1)), 'nu' : np.random.normal(0, 1, (T, 1)), 'xi' : np.random.normal(0, 1, (T, N)), 'F' : np.random.normal(0, 1, (T, r[5])), 'lambda' : (1 / np.sqrt(N)) * np.ones((r[5], N))}

	}

	return(params)

def gen_sim(sim_params, dgp):

	params = gen_params(sim_params)

	F = params[dgp]['F']
	Lambda = params[dgp]['lambda']
	theta = params[dgp]['theta']
	nu = params[dgp]['nu']
	xi = params[dgp]['xi']

	sim_output = {'y' : np.matmul(F, theta) + nu, 'X' : np.matmul(F, Lambda) + xi}

	return(sim_output)


out = gen_sim(sim_params, 'DGP1')


m = sm.OLS(out['y'], out['X']).fit()

	