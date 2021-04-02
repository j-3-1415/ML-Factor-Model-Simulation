from CODE.DataSim import *

out = gen_sim(sim_params, 'DGP1')

# Methods to cover:
# PCA, PLS, Ridge, LF, LASSO,

# run model function notes:
# Input simulated data, choose model, select penalty parameter choice method and rmax (maximum number of factors)
# Returns dictionary with
# 1. Value of penalty selection problem -- key: !!depending on method!!
# 2. Estimated parameters delta -- key: delta
# 3. Fitted values y_hat -- key: y_hat
# 4. Optimal penalty parameter -- key: !!depending on model!!

def cv(data, method):

    data = data.copy(deep=True)

    T = data['T']
    M_t = data['M_t']
    y = data['y']
    errors = data['errors']
    r = data['r']

    M_ty = M_t @ y
    trM_t = np.trace(M_t)
    error_norm = np.sum(np.power(errors, 2))
    sigma_e = np.var(errors)

    if method == 'GCV':

        numer = (1 / T) * error_norm
        denom = np.power(1 - ((1 / T) * trM_t), 2)
        crit = numer/denom

    elif method == 'Mallow':

        crit = ((1 / T) * error_norm) + (2 * sigma_e * (1 / T) * trM_t)

    elif method == 'AIC':

        crit = 2 * (r - np.log(error_norm))

    elif method == 'BIC':

        crit = (T * r) - (2 * np.log(error_norm))

    return(crit)


def run_model(data, model, method, rmax):
    X = data['X']
    Y = data['y']
    T = X.shape[0]
    N = X.shape[1]

    S_xx = (X.T @ X) / T
    S_xy = (X.T @ Y) / T
    I = np.identity(N)

    psi_svd, sigma, phi_T_svd = np.linalg.svd(X)
    psi_svd, sigma, phi_T_svd = psi_svd.real, sigma.real, phi_T_svd.real

    if model == 'PC':
        if method == 'GCV':
            GCVs = []
            for k in range(rmax):
                psi_a = psi_svd[:, :k]
                delta_pc = np.linalg.inv(psi_a.T @ psi_a) @ psi_a.T @ Y
                M_ty = psi_a @ delta_pc
                GCV = 1 / T * np.linalg.norm(Y - M_ty) ** 2 / (1 - 1 / T * k)
                GCVs.append({'k': k, 'GCV': GCV, 'delta': delta_pc, 'y_hat': M_ty})
            opt = min(GCVs, key=lambda x: x['GCV'])
            print('Model: PC, Method: GCV')
            print('GCV = ' + str(opt['GCV']) + ' --> k = ' + str(opt['k']))
            return opt

        if method == 'Mallow':
            Mallows = []
            for k in range(rmax):
                psi_a = psi_svd[:, :k]
                delta_pc = np.linalg.inv(psi_a.T @ psi_a) @ psi_a.T @ Y
                M_ty = psi_a @ delta_pc
                sigma = np.var(Y - M_ty)  # consistent estimator of variance = variance of residuals
                Mallow = (1/T) * np.linalg.norm(Y - M_ty) ** 2 + 2 * sigma * (1/T) * k
                Mallows.append({'k': k, 'Mallow': Mallow, 'delta': delta_pc, 'y_hat': M_ty})
            opt = min(Mallows, key=lambda x: x['Mallow'])
            print('Model: PC, Method: Mallow')
            print('Mallow = ' + str(opt['Mallow']) + ' --> k = ' + str(opt['k']))
            return opt

    if model == 'Ridge':
        if method == 'GCV':
            GCVs = []
            for alpha in np.linspace(0, 1, 100): # change this!!!
                delta_ridge = np.linalg.inv(S_xx + alpha * I) @ S_xy
                M_ty = X @ delta_ridge
                M_t = X @ np.linalg.inv(S_xx + alpha * I) @ (X.T/T)
                GCV = 1 / T * np.linalg.norm(Y - M_ty) ** 2 / (1 - 1 / T * np.trace(M_t))
                GCVs.append({'GCV': GCV, 'delta': delta_ridge, 'y_hat': M_ty, 'alpha': alpha})
            opt = min(GCVs, key=lambda x: x['GCV'])
            print('Model: Ridge, Method: GCV')
            print('Optimal alpha = ' + str(opt['alpha']) + ' with GCV = ' + str(opt['GCV']))
            return opt

        if method == 'Mallow':
            Mallows = []
            for alpha in np.linspace(0, 1, 100): # change this!!!
                delta_ridge = np.linalg.inv(S_xx + alpha * I) @ S_xy
                M_ty = X @ delta_ridge
                M_t = X @ np.linalg.inv(S_xx + alpha * I) @ (X.T/T)
                sigma = np.var(Y - M_ty)  # consistent estimator of variance = variance of residuals
                Mallow = (1/T) * np.linalg.norm(Y - M_ty) ** 2 + 2 * sigma * (1/T) * np.trace(M_t)
                Mallows.append({'Mallow': Mallow, 'delta': delta_ridge, 'y_hat': M_ty, 'alpha': alpha})
            opt = min(Mallows, key=lambda x: x['Mallow'])
            print('Model: Ridge, Method: Mallow')
            print('Optimal alpha = ' + str(opt['alpha']) + ' with Mallow = ' + str(opt['Mallow']))
            return opt




mods = run_model(out, 'PC', 'GCV', rmax=10)
mods = run_model(out, 'PC', 'Mallow', rmax=10)
mods = run_model(out, 'Ridge', 'GCV', rmax=10)
mods = run_model(out, 'Ridge', 'Mallow', rmax=10)



################### LEGACY/EXPERIMENT CODE #####################

# legacy method with straight forward eigenvectors for PCA (no SVD)
S_xx = np.matmul(out['X'], out['X'].T)/200
lambda_sq, psi = np.linalg.eig(S_xx)
lambda_sq = lambda_sq.real
psi = psi.real

alpha = lambda_sq[10]

num_vals = len([i for i in lambda_sq if i >= alpha])
new_psi = psi[:, :num_vals]

delta_hat = np.linalg.inv(new_psi.T @ new_psi) @ new_psi.T @ out['y']

[np.absolute(i) for i in lambda_sq][:10]  # check magnitude of complex part


# cross checking the equivalence between eigenvectors and SVD

X = out['X']
T = X.shape[0]
S_xx = (X.T @ X) / T
S_xx_2 = (X @ X.T) / T

psi_svd, sigma, phi_T_svd = np.linalg.svd(X)

lamb_N, phi_eig = np.linalg.eig(S_xx)
lamb_T, psi_eig = np.linalg.eig(S_xx_2)

psi_svd, sigma, phi_T_svd, lamb_N, phi_eig, lamb_T, psi_eig = psi_svd.real, sigma.real, phi_T_svd.real, lamb_N.real, phi_eig.real, lamb_T.real, psi_eig.real

phi_eig = phi_eig[:, np.argsort(-1 * lamb_N)]
psi_eig = psi_eig[:, np.argsort(-1 * lamb_T)]

lamb_N = lamb_N[np.argsort(-1 * lamb_N)]
lamb_T = lamb_T[np.argsort(-1 * lamb_T)]

# psi_j = (X @ phi_eig[:, 0]) / sigma[0]

alpha = 100
psi_a = psi_svd[:, list(np.where(sigma >= alpha))]
psi_a = psi_a.reshape((psi_a.shape[0], psi_a.shape[2]))
delta_pc_a = np.linalg.inv(psi_a.T @ psi_a) @ psi_a.T @ out['y']



