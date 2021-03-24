from CODE.DataSim import *

df = gen_sim(sim_params, 'DGP1')

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

