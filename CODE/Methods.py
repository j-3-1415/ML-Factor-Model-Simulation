# from CODE.DataSim import *
from DataSim import *
import operator
import pandas as pd

# Methods to cover:
# PCA, PLS, Ridge, LF, LASSO,

# run model function notes:
# Input simulated data, choose model, select penalty parameter choice method and rmax (maximum number of factors)
# Returns dictionary with
# 1. Value of penalty selection problem -- key: !!depending on method!!
# 2. Estimated parameters delta -- key: delta
# 3. Fitted values y_hat -- key: y_hat
# 4. Optimal penalty parameter -- key: !!depending on model!!

# alternatively: start from equation 14
def get_model_output(data, model, iteration):

	data = data.copy()

    X = data['X']
    y = data['y']
    T = X.shape[0]
    N = X.shape[1]
    r_max = data['r_max']

    S_xx = (X.T @ X) / T
    S_xxT = (X @ X.T) / T
    S_xy = (X.T @ y) / T
    I = np.identity(S_xx.shape[0])

    psi_svd, sigma, phi_T_svd = np.linalg.svd(X)
    psi_svd, sigma, phi_T_svd = psi_svd.real, sigma.real, phi_T_svd.real

    lambda_sq, psi = np.linalg.eig(S_xx)
    lambda2, psi2 = np.linalg.eig(S_xxT)
    lambda_sq, psi = lambda_sq.real, psi.real

    if model == 'PC':
    	psi_svd = psi_svd[:, :data['k'][iteration]]
    	M_t = psi_svd @ np.linalg.inv(psi_svd.T @ psi_svd) @ psi_svd.T

    elif model == 'Ridge':
    	q = lambda_sq / (lambda_sq + data['alphas'][iteration])
    	q = q.reshape(1, q.shape[0])
    	M_t = psi_svd @ np.linalg.inv(psi_svd.T @ psi_svd) @ psi_svd.T
    	M_t = np.multiply(q, M_t[:, :, np.newaxis]).sum(axis=2)

    elif model == 'LF':
    	alpha = data['alphas'][iteration]
        d = 0.018 / np.amax(lambda_sq)  # as defined on page 327
        q = 1 - np.power((1 - d * lambda_sq), (1 / alpha))
        q = (q / lambda_sq).reshape(1, q.shape[0])
        M_t = psi_svd @ psi_svd.T
        M_t = X @ X.T @ np.multiply(q, M_t[:, :, np.newaxis]).sum(axis=2)

    elif model == 'PLS':
    	V_k = np.tile(X.T @ y, (1, data['k'][iteration]))
    	for i in range(1, data['k'][iteration]):
    		V_k[:, i] = np.power(X.T @ X, i) @ V_k[:, i]
    	M_t = X @ V_k @ np.linalg.inv(V_k.T @ X.T @ X @ V_k) @ V_k.T @ X.T

    elif model == 'BaiNg':

    	F = np.sqrt(T) * psi2[:, :data['k'][iteration]]
    	lamb = (1 / T) * (F.T @ X)
    	V = 0
    	for i in range(N):
    		for t in range(T):
    			V += np.power(X[t, i] - (lamb[:, i] @ F[t, :]), 2)
    	V = (1 / (N * T)) * V
    	PC = V + (data['k'][iteration] * (2 / T))
    	data['criteria'][iteration] = PC
    	return(data)

    data['M_t'] = M_t
    data['M_ty'] = M_t @ y

    return(data)

def cv(data, model, method, iteration):
    data = data.copy()

    data = get_model_output(data, model, iteration)

    if model == 'BaiNg':
    	return(data)

    M_t = data['M_t']
	M_ty = data['M_ty']
    y = data['y']
    errors = y - M_ty

    if model in ['PC', 'PLS']:
    	r = data['k'][iteration]

    T = y.shape[0]

    trM_t = np.trace(M_t)
    error_norm = np.sum(np.power(errors, 2))
    sigma_e = np.var(errors)

    if method == 'GCV':

        numer = (1 / T) * error_norm
        denom = np.power(1 - ((1 / T) * trM_t), 2)
        crit = numer / denom

    elif method == 'Mallow':

        crit = ((1 / T) * error_norm) + (2 * sigma_e * (1 / T) * trM_t)

    elif method == 'AIC':

        crit = 2 * (r - np.log(error_norm))

    elif method == 'BIC':

        crit = -1 * ((T * r) - (2 * np.log(error_norm)))

    elif method == 'LOO_CV':
    	diag = 1 - np.diag(M_t)
    	crit = (1 / T) * np.sum([(errors[i] / diag[i])**2 for i in range(len(errors))])

    data['criteria'][iteration] = crit

    return(data)

def monte_carlo(monte_params, sim_params):

	N = monte_params['N']
	T = monte_params['T']
	sims = monte_params['sims']
	DGP = monte_params['DGP']
	best = []
	model = monte_params['model']
	eval_form = monte_params['eval']
	crit_var = ['alphas', 'k'][model in ['PC', 'PLS', 'BaiNg']]
	r_max = sim_params['r_max'][DGP-1]

	crit_dict = {'k' : 
		{'PC' : [list(range(0, (r_max + 1)))] * 6,
		'PLS' : [list(range(1, (r_max + 1)))] * 6,
		'BaiNg' : [list(range(0, (r_max + 1)))] * 6},
		'alphas' :
		{'Ridge' : [N * np.arange(0, 0.1, 0.001), 
					N * np.arange(0, 0.01, 0.0001),
					N * np.arange(0, 0.1, 0.0005),
					N * np.arange(0, 0.1, 0.001),
					N * np.arange(0, 0.15, 0.001),
					N * np.arange(0, 0.1, 0.001)],
		'LF' : [N * np.arange(0.000001, 0.0003, 0.00002),
				N * np.arange(0.00001, 0.0002, 0.00001),
				N * np.arange(0.000001, 0.00005, 0.0000025),
				N * np.arange(0.000001, 0.0004, 0.00001),
				N * np.arange(0.000001, 0.0004, 0.00002),
				N * np.arange(0.000001, 0.016, 0.001)]}
		}

	header = "Simulation|Current Best|Rolling Mean|Rolling Std"
	cols = [len(x) for x in header.split("|")]
	print(header)
	for sim in range(sims):

		sim_data = gen_sim(sim_params, 'DGP' + str(DGP), N, T)
		sim_data[crit_var] = crit_dict[crit_var][model][DGP-1]
		sim_data['criteria'] = np.ones(len(sim_data[crit_var]))

		for i in range(len(sim_data[crit_var])):

			sim_data = cv(sim_data, model, eval_form, i)

		best_dict = dict(zip(sim_data[crit_var], sim_data['criteria']))
		curr_best = min(best_dict.items(), key=operator.itemgetter(1))[0]
		best.append(curr_best)

		print("-" * len(header))
		print("|" + str(sim + 1) + " " * (cols[0] - len(str(sim + 1)) - 1) + "|" + 
			'{0:.2f}'.format(curr_best) + " " * (cols[1] - len('{0:.2f}'.format(curr_best))) + "|" + 
			'{0:.2f}'.format(np.mean(best)) + " " * (cols[2] - len('{0:.2f}'.format(np.mean(best)))) + "|" + 
			'{0:.2f}'.format(np.std(best)) + " " * (cols[3] - len('{0:.2f}'.format(np.std(best))) - 1) + "|")
	print("-" * len(header))

monte_params = {
	'N' : 100,
	'T' : 50,
	'sims' : 25,
	'DGP' : 1,
	'model' : 'BaiNg',
	'eval' : 'LOO_CV'
}

monte_carlo(monte_params, sim_params)


sizes = ['500x200', '100x50']
models = ['PC', 'PLS', 'Ridge', 'LF', 'BaiNg']
evals = ['GCV', 'Mallow', 'AIC', 'BIC', 'LOO_CV']
comb = [[i, j, k] for i in sizes for j in models for k in evals]
check = pd.DataFrame(comb)
check.columns = ['TxN', 'Model', 'Eval']
check.loc[:, ['DGP' + str(i+1) for i in range(6)]] = 'No'
check = check[~((check['Model']!='PLS')&(check['Eval'] == 'LOO_CV'))].reset_index().drop('index', axis=1)
check.loc[check['Model'] == 'BaiNg', 'Eval'] = 'BaiNg'
check = check[~check.duplicated()].reset_index().drop('index', axis=1)


