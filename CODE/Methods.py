from CODE.DataSim import *

out = gen_sim(sim_params, 'DGP1')

# Methods to cover:
# PCA, PLS, Ridge, LF, LASSO,

S_xx = np.matmul(out['X'], out['X'].T)/200
lambda_sq, psi = np.linalg.eig(S_xx)
lambda_sq = lambda_sq.real
psi = psi.real

alpha = lambda_sq[10]

num_vals = len([i for i in lambda_sq if i >= alpha])
new_psi = psi[:, :num_vals]

delta_hat = np.linalg.inv(new_psi.T @ new_psi) @ new_psi.T @ out['y']

[np.absolute(i) for i in lambda_sq][:10]  # check magnitude of complex part

# for now: phi = psi

# M_t * y = y_hat

# what is lambda without sq?

X = out['X']
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


def get_alpha(data, model, rmax):
    X = data['X']
    Y = data['y']
    T = X.shape[0]

    psi_svd, sigma, phi_T_svd = np.linalg.svd(X)
    psi_svd, sigma, phi_T_svd = psi_svd.real, sigma.real, phi_T_svd.real

    if model == 'PC':
        GCVs = []
        for k in range(rmax):
            psi_a = psi_svd[:, :k]
            delta_pc = np.linalg.inv(psi_a.T @ psi_a) @ psi_a.T @ Y
            M_ty = psi_a @ delta_pc
            GCV = 1 / T * np.linalg.norm(Y - M_ty) ** 2 / (1 - 1 / T * k)
            GCVs.append((k, GCV))
        return GCVs

get_alpha(out, 'PC', rmax = 10)

#  optimal k is 4!!!

