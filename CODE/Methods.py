from CODE.DataSim import *
# from DataSim import *
import operator
import pandas as pd
from CODE.OutputTex import out_latex
import os


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

	r_max = int(data['r_max'])

	X = data['X']
	# X = X[:, :r_max]
	y = data['y']
	T = X.shape[0]
	N = X.shape[1]
	# r_max = min(N, T)

	S_xx = (X.T @ X) / T
	S_xy = (X.T @ y) / T

	# psi_svd, sigma, phi_T_svd = np.linalg.svd(X)
	# psi_svd, sigma, phi_T_svd = psi_svd.real, sigma.real, phi_T_svd.real

	sigma, psi_svd = np.linalg.eig((X @ X.T)/ T)
	sigma, psi_svd = sigma.real, psi_svd.real

	lambda_sq, psi = np.linalg.eig(S_xx)
	lambda_sq, psi = lambda_sq.real, psi.real

	if model == 'PC':
		psi_svd = psi_svd[:, :data['k'][iteration]]
		delt = np.linalg.inv(psi_svd.T @ psi_svd) @ psi_svd.T
		M_t = psi_svd @ delt
		M_ty = M_t @ y
		delta = delt @ y

	elif model == 'Ridge':
		I = np.identity(X.shape[1])
		delt = (np.linalg.inv(S_xx + data['alphas'][iteration] * I) @ X.T) / T
		M_t = X @ delt
		M_ty = M_t @ y
		delta = delt @ y

	elif model == 'LF':
		alpha = data['alphas'][iteration]
		d = 0.018 / np.amax(lambda_sq)  # as defined on page 327
		delt = np.zeros((N, T))
		for i in range(0, min(N, T)):
			q = (1 - np.power((1 - d * lambda_sq[i]), (1 / alpha))) / lambda_sq[i]
			psi_j = psi_svd[:, i].reshape((T, 1))
			delt += q * (X.T @ psi_j @ psi_j.T)
		delt = delt / T
		M_t = X @ delt
		M_ty = M_t @ y
		delta = delt @ y

	elif model == 'PLS':
		# vals, vecs = np.linalg.eig(S_xy @ S_xy.T)
		# vals, vecs = vals.real, vecs.real
		# vecs = vecs[:, np.argsort(vals)[::-1]]
		# V_k = vecs[:, :data['k'][iteration]]
		# u, s, v = np.linalg.svd(X.T @ y @ y.T @ X)
		# u, s, v = u.real, s.real, v.real
		# u = u[:, np.argsort(s)[::-1]]
		# V_k = u[:, :data['k'][iteration]]
		# V_k = np.tile(X.T @ y, (1, data['k'][iteration]))
		# for i in range(1, data['k'][iteration]):
		#     V_k[:, i] = X.T @ X @ V_k[:, i - 1]
		# delt = V_k @ np.linalg.inv(V_k.T @ X.T @ X @ V_k) @ V_k.T @ X.T
		# M_t = X @ delt
		# M_ty = M_t @ y
		# delta = delt @ y
		k = data['k'][iteration]
		# S = S_xy
		S = (X.T @ y).reshape((S_xy.shape[0], 1))
		R = np.zeros((N, k))
		T = np.zeros((T, k))
		P = np.zeros((N, k))
		for i in range(k):
			if i == 0:
				u, s, v = np.linalg.svd(S)
				u, s, v = u.real, s.real, v.real
			else:
				p = P[:, i-1].reshape((N, 1))
				newS = S-(p @ np.linalg.inv(p.T @ p) @ p.T @ S)
				u, s, v = np.linalg.svd(newS)
				u, s, v = u.real, s.real, v.real
			R[:, i] = u[:, i]
			T[:, i] = X @ R[:, i]
			P[:, i] = (X.T @ T[:, i]) / (T[:, i].T @ T[:, i])

		delt = R @ np.linalg.inv(T.T @ T) @ T.T
		M_t = X @ delt
		M_ty = M_t @ y
		delta = delt @ y


	elif model == 'BaiNg':
		F_tild = np.sqrt(T) * psi_svd[:, :data['k'][iteration]]
		lambda_tild = np.linalg.inv(F_tild.T @ F_tild) @ (F_tild.T @ X)
		M_t = F_tild @ lambda_tild
		M_ty = None

	data['M_t'] = M_t
	data['M_ty'] = M_ty
	data['delta'] = delta

	return(data)

def mallow_sig(data, model):
	data = data.copy()

	# r_max = int(data['r_max'])

	X = data['X']
	y = data['y']
	T = X.shape[0]
	N = X.shape[1]

	r_max = min(N, T)

	psi, sigma, phi_T_svd = np.linalg.svd(X)
	psi, sigma, phi_T_svd = psi.real, sigma.real, phi_T_svd.real

	F_tild = np.sqrt(T) * psi[:, :r_max]
	lambda_tild = np.linalg.inv(F_tild.T @ F_tild) @ (F_tild.T @ X)

	e = X - (F_tild @ lambda_tild)
	sigma_e = (1 / (N * T)) * np.sum(np.power(e, 2))

	return(e, sigma_e)


def cv(data, model, method, iteration):
	data = data.copy()

	data = get_model_output(data, model, iteration)

	y = data['y']
	N = data['X'].shape[1]
	T = y.shape[0]
	e, sigma_e = mallow_sig(data, model)

	if model == 'BaiNg':
		V = (1 / (N * T)) * np.sum(np.power(data['X'] - data['M_t'], 2))
		crit =  V + (sigma_e * data['k'][iteration] * ((N + T) / (N * T)) * np.log(min(N, T)))
		# crit = V * (1 + data['k'][iteration] * ((N + T) / (N * T)) * np.log(min(N, T)))
		data['criteria'][iteration] = crit
		data['MSE'][iteration] = None
		return(data)

	M_t = data['M_t']
	M_ty = data['M_ty']
	errors = y - M_ty

	trM_t = np.trace(M_t)
	error_norm = np.sum(np.power(errors, 2))

	if method == 'GCV':

		numer = (1 / T) * error_norm
		denom = np.power(1 - ((1 / T) * trM_t), 2)
		crit = numer / denom

	elif method == 'Mallow':
		crit = ((1 / T) * error_norm) + (2 * sigma_e * (1 / T) * trM_t)

	elif method == 'AIC':
		crit = 2 * (trM_t - np.log(error_norm))

	elif method == 'BIC':
		crit = 1 * ((T * trM_t) - (2 * np.log(error_norm)))

	elif method == 'LOO_CV':
		crit = (1 / T) * np.sum(np.power(errors / (1 - np.diag(M_t)), 2))

	data['criteria'][iteration] = crit
	data['MSE'][iteration] = error_norm / len(errors)
	data['DOF'][iteration] = trM_t

	return (data)


def monte_carlo(monte_params):
	N = monte_params['N']
	T = monte_params['T']
	sims = monte_params['sims']
	DGP = monte_params['DGP']
	best_param = []
	best_MSE = []
	best_DOF = []
	model = monte_params['model']
	eval_form = monte_params['method']
	crit_var = ['alphas', 'k'][model in ['PC', 'PLS', 'BaiNg']]

	sim_params = {
		'r': [4, 50, 5, 5, N, 1],
		'r_max': [14, np.floor(min(N, (T / 2))), np.floor(min(15, min(N, (T / 2)))), 15, np.floor(min(N, (T / 2))), 11]
	}
	r_max = int(sim_params['r_max'][DGP - 1])
	# r_max = min(N, T)

	crit_dict = {'k':
		{'PC': [list(range(0, (r_max + 1)))] * 6,
			'PLS': [list(range(1, (r_max + 1)))] * 6,
			'BaiNg': [list(range(0, (r_max + 1)))] * 6},
		'alphas':
		{'Ridge': [N * np.arange(0, 0.1, 0.001),
			N * np.arange(0, 0.01, 0.0001),
			N * np.arange(0, 0.1, 0.0005),
			N * np.arange(0, 0.1, 0.001),
			N * np.arange(0, 0.15, 0.001),
			N * np.arange(0, 0.1, 0.001)],
		'LF': [N * np.arange(0.000001, 0.0003, 0.00002),
			N * np.arange(0.00001, 0.0002, 0.00001),
			N * np.arange(0.000001, 0.00005, 0.0000025),
			N * np.arange(0.000001, 0.0004, 0.00001),
			N * np.arange(0.000001, 0.0004, 0.00002),
			N * np.arange(0.000001, 0.016, 0.001)]}
			}

	header = "Simulation|Current Best|Roll Mean|Roll Std|Roll DOF|Roll MSE"
	cols = [len(x) for x in header.split("|")]
	print(header)
	for sim in range(sims):

		sim_data = gen_sim(sim_params, 'DGP' + str(DGP), N, T)
		sim_data[crit_var] = crit_dict[crit_var][model][DGP - 1]
		sim_data['criteria'] = np.ones(len(sim_data[crit_var]))
		sim_data['MSE'] = np.ones(len(sim_data[crit_var]))
		sim_data['DOF'] = np.ones(len(sim_data[crit_var]))

		for i in range(len(sim_data[crit_var])):
			sim_data = cv(sim_data, model, eval_form, i)
			# return(sim_data)

		param_dict = dict(zip(sim_data[crit_var], sim_data['criteria']))
		MSE_dict = dict(zip(sim_data[crit_var], sim_data['MSE']))
		DOF_dict = dict(zip(sim_data[crit_var], sim_data['DOF']))
		curr_crit = min(param_dict.items(), key=operator.itemgetter(1))[0]
		curr_MSE = MSE_dict[curr_crit]
		curr_DOF = DOF_dict[curr_crit]
		best_param.append(curr_crit)
		best_MSE.append(curr_MSE)
		best_DOF.append(curr_DOF)

		print("-" * len(header))
		print("|" + str(sim + 1) + " " * (cols[0] - len(str(sim + 1)) - 1) + "|" +
			'{0:.2f}'.format(curr_crit) + " " * (cols[1] - len('{0:.2f}'.format(curr_crit))) + "|" +
			'{0:.2f}'.format(np.mean(best_param)) + " " * (cols[2] - len('{0:.2f}'.format(np.mean(best_param)))) + "|" +
			'{0:.2f}'.format(np.std(best_param)) + " " * (cols[3] - len('{0:.2f}'.format(np.std(best_param)))) + "|" +
			'{0:.2f}'.format(np.mean(best_DOF)) + " " * (cols[4] - len('{0:.2f}'.format(np.mean(best_DOF)))) + "|" + 
			'{0:.2f}'.format(np.mean(best_MSE)) + " " * (cols[5] - len('{0:.2f}'.format(np.mean(best_MSE))) - 1) + "|")
	print("-" * len(header))

	return(best_param, best_MSE, best_DOF)


# monte_params = {
# 	'N': 100,
# 	'T': 50,
# 	'sims': 25,
# 	'DGP': 1,
# 	'model': 'PLS',
# 	'method': 'LOO_CV'
# }

# best_param, best_MSE, best_DOF = monte_carlo(monte_params)

def gen_tex_dict(tex_params):

	tex_params = tex_params.copy()

	tex_dict = {'N' : tex_params['N'], 'T' : tex_params['T'],
		'method' : tex_params['method'],
		'r' : [4, 50, 5, 5, tex_params['N'], 1],
		'PC' : {'params' : np.ones(6), 'se' : np.ones(6)},
		'PLS' : {'params' : np.ones(6), 'se' : np.ones(6)},
		'Ridge' : {'alpha' : {'params' : np.ones(6), 'se' : np.ones(6)},
			'DOF' : {'params' : np.ones(6), 'se' : np.ones(6)}},
		'LF' : {'alpha' : {'params' : np.ones(6), 'se' : np.ones(6)},
			'DOF' : {'params' : np.ones(6), 'se' : np.ones(6)}}
		}

	for model in tex_params['models']:
		for dgp in range(6):
			print("Running Model Simulation %s"%model)
			print('Running DGP%s'%str(dgp + 1))
			monte_params = {
				'N': tex_params['N'],
				'T': tex_params['T'],
				'sims': tex_params['sims'],
				'DGP': (dgp + 1),
				'model': model,
				'method': tex_params['method']
			}
			if (model == 'PLS')&(tex_params['method'] == 'GCV'):
				monte_params['method'] = 'LOO_CV'

			params, mse, dof = monte_carlo(monte_params)

			if model in ['PC', 'PLS']:
				tex_dict[model]['params'][dgp] = np.mean(params)
				tex_dict[model]['se'][dgp] = np.std(params)
			else:
				tex_dict[model]['alpha']['params'][dgp] = np.mean(params)
				tex_dict[model]['alpha']['se'][dgp] = np.std(params)
				tex_dict[model]['DOF']['params'][dgp] = np.mean(dof)
				tex_dict[model]['DOF']['se'][dgp] = np.std(dof)

	return(tex_dict)

for N, T in [(200, 500), (100, 50)]:
	for method in ['GCV', 'Mallow']:

		tex_params = {
			'N': N,
			'T': T,
			'sims': 100,
			'method': method,
			'models' : ['PC', 'PLS', 'Ridge', 'LF']
		}

		file = "Table_N%s_T%s_Eval%s_Sims%s.tex"%(N, T, method, 100)
		file = os.path.abspath("..") + "/Report/" + file
		tex_string = out_latex(file, gen_tex_dict(tex_params))

# sizes = ['500x200', '100x50']
# models = ['PC', 'PLS', 'Ridge', 'LF', 'BaiNg']
# evals = ['GCV', 'Mallow', 'AIC', 'BIC', 'LOO_CV']
# comb = [[i, j, k] for i in sizes for j in models for k in evals]
# check = pd.DataFrame(comb)
# check.columns = ['TxN', 'Model', 'Eval']
# check.loc[:, ['DGP' + str(i + 1) for i in range(6)]] = 'No'
# check = check[~((check['Model'] != 'PLS') & (check['Eval'] == 'LOO_CV'))].reset_index().drop('index', axis=1)
# check.loc[check['Model'] == 'BaiNg', 'Eval'] = 'BaiNg'
# check = check[~check.duplicated()].reset_index().drop('index', axis=1)
# check[(check['Model'] == 'PC') & (check['Eval'] == 'GCV')][['DGP1', 'DGP3', 'DGP4']]



