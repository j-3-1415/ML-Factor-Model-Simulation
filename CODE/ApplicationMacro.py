from CODE.Methods import *

from FredMD import FredMD
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os

outpath = os.getcwd() + '/OUTPUT/'


def get_best_model(data, model, eval_form, r_max):
    data = data.copy()

    T, N = data['X'].shape
    data['r_max'] = r_max
    crit_var = ['alphas', 'k'][model in ['PC', 'PLS', 'BaiNg']]

    crit_dict = {'k':
                     {'PC': list(range(1, (r_max + 1))),
                      'PLS': list(range(1, (r_max + 1))),
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
    DOF_dict = dict(zip(data[crit_var], data['DOF']))
    curr_par = min(param_dict.items(), key=operator.itemgetter(1))[0]
    curr_MSE = min(MSE_dict.values())
    curr_DOF = DOF_dict[curr_par]

    if model in ['Ridge', 'LF']:
        data['alphas'] = [curr_par]

    elif model in ['PC', 'PLS']:
        data['k'] = [curr_par]

    mat = get_model_output(data, model, 0)
    delta = mat['delta']


    return (curr_par, curr_MSE, curr_DOF, delta)


# putting everything together
def apply_model(target, horizon, train_size, model, eval_form, r_max, fmd):

    fmd.apply_transforms()
    # fmd.factor_standardizer_method(code=2)
    df = fmd.rawseries

    df['y'] = fmd.rawseries[target]# .tail(-2) # in clean data we drop first two obs

    # because of many columns with NAs remove columns where more than 1% are missing
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

    par, MSE, DOF, delta = get_best_model(data_train, model, eval_form, r_max)

    if model in ['Ridge', 'PLS', 'LF']:
        pred_oos = X_test @ delta

    elif model == 'PC':
        T, N = X_test.shape
        psi_svd, sigma, phi_T_svd = np.linalg.svd(X_test @ X_test.T / T)
        psi_svd, sigma, phi_T_svd = psi_svd.real, sigma.real, phi_T_svd.real
        psi_svd = psi_svd[:, :par]

        pred_oos = psi_svd @ delta


    errors = Y_test - pred_oos
    MSE_oos = np.power(np.linalg.norm(errors), 2) / len(errors)

    return par, MSE, MSE_oos, DOF

def get_plot_data(params, fmd, h):
    MSEs = []

    for model in ['PC', 'Ridge', 'PLS', 'LF']:    # leave LF out for now
        for y in ['INDPRO', 'UNRATE', 'HOUST']:
            for crit in ['GCV', 'Mallow']:
                params_new = params.copy()
                params_new['model'] = model
                params_new['target'] = y
                params_new['eval_form'] = crit
                params_new['horizon'] = h
                par, MSE_is, MSE_oos, DOF = apply_model(**params_new, fmd=fmd)
                MSEs.append(
                    {
                        'Target': y,
                        'Method': model + ': ' + crit,
                        'MSE_is': MSE_is,
                        'MSE_oos': MSE_oos,
                        'opt_par': par,
                        'horizon': h,
                        'DOF': DOF
                    }
                )
    MSE_df = pd.DataFrame(MSEs)
    return(MSE_df)

def plot_mse(plot_df, mse_type, plotname):
    fig, ax = plt.subplots()
    sns.scatterplot(data=plot_df, x='Method', y=mse_type, s=40, hue='Target', ax = ax)
    # for i in range(len(plot_df['Method'])):
    #     plt.annotate(round(plot_df[mse_type][i], 4), (plot_df['Method'][i], plot_df[mse_type][i] + np.std(plot_df[mse_type])),
    #                  ha='center')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    plt.legend(loc='upper left')
    plt.ylim((-0.005, max(plot_df[mse_type] + 0.01)))
    plt.ylabel('MSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outpath + mse_type + plotname + '.png')
    plt.close()




# # variables of interest: Industrial Production (INDPRO), Unemployment Rate (UNRATE), (HOUST) following
# # Coulombe et al.

fmd = FredMD(Nfactor=None, vintage=None, maxfactor=8, standard_method=2, ic_method=2)

params = {
    'model': 'Ridge',
    'eval_form': 'Mallow',
    'r_max': 15,
    'target': 'HOUST',
    'horizon': 1,
    'train_size': 0.8}

par, MSE_is, MSE_oos = apply_model(**params, fmd=fmd)

# horizon = 1
df_p_h1 = get_plot_data(params, fmd, h=1)
plot_mse(df_p_h1, mse_type='MSE_oos', plotname='_h1')

# remove houst for better comparison
df_p_h1_sub = df_p_h1[df_p_h1['Target'] != 'HOUST']
plot_mse(df_p_h1_sub, mse_type='MSE_oos', plotname='_h1_noHOUST')

# horizon = 3
df_p_h3 = get_plot_data(params, fmd, h=3)
plot_mse(df_p_h3, mse_type='MSE_oos', plotname='_h3')

# remove houst for better comparison
df_p_h3_sub = df_p_h3[df_p_h3['Target'] != 'HOUST']
plot_mse(df_p_h3_sub, mse_type='MSE_oos', plotname='_h3_noHOUST')

# horizon = 9
df_p_h9 = get_plot_data(params, fmd, h=9)
plot_mse(df_p_h9, mse_type='MSE_oos', plotname='_h9')

# remove houst for better comparison
df_p_h9_sub = df_p_h9[df_p_h9['Target'] != 'HOUST']
plot_mse(df_p_h9_sub, mse_type='MSE_oos', plotname='_h9_noHOUST')


# export table to latex
df = df_p_h1.sort_values(by=['Target', 'Method'])
df['opt_par'] = round(df['opt_par'], 4)
df['MSE_oos3'] = df_p_h3['MSE_oos']
df['MSE_oos9'] = df_p_h9['MSE_oos']

print(df[df['Target']=='INDPRO'].to_latex(index=False, columns=['Method', 'MSE_oos', 'MSE_oos3', 'MSE_oos9', 'opt_par', 'DOF']))
print(df[df['Target']=='UNRATE'].to_latex(index=False, columns=['Method', 'MSE_oos', 'MSE_oos3', 'MSE_oos9', 'opt_par', 'DOF']))
print(df[df['Target']=='HOUST'].to_latex(index=False, columns=['Method', 'MSE_oos', 'MSE_oos3', 'MSE_oos9', 'opt_par', 'DOF']))




