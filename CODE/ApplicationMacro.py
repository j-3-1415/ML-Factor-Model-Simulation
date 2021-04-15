from CODE.Methods import *

from FredMD import FredMD
from sklearn.preprocessing import StandardScaler


def get_best_model(data, model, eval_form, r_max):
    data = data.copy()

    T, N = data['X'].shape
    data['r_max'] = r_max
    crit_var = ['alphas', 'k'][model in ['PC', 'PLS', 'BaiNg']]

    crit_dict = {'k':
                     {'PC': list(range(1, (r_max + 1))),
                      'PLS': list(range(0, (r_max + 1))),
                      'BaiNg': list(range(0, (r_max + 1)))},
                 'alphas':
                     {'Ridge': N * np.arange(0, 0.1, 0.001),
                      'LF': N * np.arange(0.000001, 0.0003, 0.00002)}
                 }

    data[crit_var] = crit_dict[crit_var][model]

    data['criteria'] = np.ones(len(data[crit_var]))
    data['MSE'] = np.ones(len(data[crit_var]))
    data['DOF'] = np.ones(len(data[crit_var]))

    for i in range(len(data[crit_var])):
        data = cv(data, model, eval_form, i)

    param_dict = dict(zip(data[crit_var], data['criteria']))
    MSE_dict = dict(zip(data[crit_var], data['MSE']))
    curr_par = min(param_dict.items(), key=operator.itemgetter(1))[0]
    curr_MSE = min(MSE_dict.values())


    return (curr_par, curr_MSE, data)


# putting everything together
def apply_model(target, horizon, train_size, model, eval_form, r_max, fmd):

    # fmd.apply_transforms()
    # fmd.factor_standardizer_method(code=2)
    df = fmd.rawseries

    df['y'] = fmd.rawseries[target]# .tail(-2) # in clean data we drop first two obs

    # because of many columns with missing NAs remove columns where more than 1% are missing
    df = df[df.columns[df.isnull().mean() < 0.01]].dropna(axis=0)
    # like this we only drop 3 observations

    scale = StandardScaler()

    train = df.head(int(len(df)*train_size))
    X_train = train[train.columns[~train.columns.isin([target, 'y'])]].tail(-horizon)
    X_train = scale.fit_transform(X_train)

    Y_train = df['y'].head(int(len(df['y'])*train_size))
    Y_train = (1 / horizon) * np.log(Y_train.shift(+horizon).dropna(axis=0) / Y_train.tail(-horizon)).to_numpy()


    test = df.tail(len(df) - len(train))
    X_test = test[test.columns[~test.columns.isin([target, 'y'])]].tail(-horizon)
    X_test = scale.fit_transform(X_test)

    Y_test = df['y'].head(len(df) - len(train))
    Y_test = (1 / horizon) * np.log(Y_test.shift(+horizon).dropna(axis=0) / Y_test.tail(-horizon)).to_numpy()


    arrays = [Y_train, X_train]
    keys = ['y', 'X']
    data_train = {k: v for k, v in zip(keys, arrays)}

    par, MSE, mat = get_best_model(data_train, model, eval_form, r_max)

    if model == 'Ridge':
        pred_oos = X_test @ mat['delta']

    elif model == 'PC':
        T, N = X_test.shape
        psi_svd, sigma, phi_T_svd = np.linalg.svd(X_test @ X_test.T / T)
        psi_svd, sigma, phi_T_svd = psi_svd.real, sigma.real, phi_T_svd.real
        psi_svd = psi_svd[:, :par]

        pred_oos = psi_svd @ mat['delta']

    elif model == 'PLS':
        pred_oos = X_test @ mat['delta']


    errors = Y_test - pred_oos
    MSE_oos = np.power(np.linalg.norm(errors), 2) / len(errors)

    return par, MSE, MSE_oos





# # variables of interest: Industrial Production (INDPRO), Unemployment Rate (UNRATE), CPI (INF) following
# # Coulombe et al.

fmd = FredMD(Nfactor=None, vintage=None, maxfactor=8, standard_method=2, ic_method=2)

params = {
    'model': 'Ridge',
    'eval_form': 'Mallow',
    'r_max': 10,
    'target': 'INDPRO',
    'horizon': 4,
    'train_size': 0.8}

par, MSE_is, MSE_oos = apply_model(**params, fmd=fmd)

# BaiNg factor estimation
fmd.estimate_factors()
f = fmd.factors
# from bai ng we get 7 factors


# problem: PC just returns r_max as optimal number of factors
# PLS returned 5 but now not working because we need to fix delta


# code to experiment with get_best_model func

# df['y'] = df['INDPRO'].shift(+1)
#
# # because of many columns with missing NAs remove columns where more than 1% are missing
# df = df[df.columns[df.isnull().mean() < 0.01]].dropna(axis=0)
# # like this we only drop 3 observations
#
# # standardize data
# scale = StandardScaler()
# X = scale.fit_transform(df[df.columns[~df.columns.isin(['INDPRO'])]])
#
# #
# Y = scale.fit_transform(df[df.columns[df.columns.isin(['y'])]])
#
#
# arrays = [Y, X]
# keys = ['y', 'X']
# data = {k: v for k, v in zip(keys, arrays)}

# par, MSE, mat = get_best_model(data, params)

y = fmd.rawseries[target].tail(-2) # in cleaned data we drop first two obs
og = y.tail(-horizon)
shifted = y.shift(+horizon).dropna(axis=0)
y = (1/horizon) * np.log(shifted/og)

