from CODE.Methods import *
import matplotlib.pyplot as plt
import seaborn as sns
import os

outpath = os.getcwd() + '/OUTPUT/'

def get_MSE(N, T, sims, DGP, model, eval):
    params = {
    'N': N,
    'T': T,
    'sims': sims,
    'DGP': DGP,
    'model': model,
    'eval': eval}

    best_param, best_MSE = monte_carlo(params)
    MSE_avg = np.mean(best_MSE)
    return MSE_avg

def plot_MSE(N, T, sims, DGP):
    MSEs = []

    for model in ['PC', 'Ridge', 'PLS']:    # leave LF out for now
        for crit in ['GCV', 'Mallow']:
            MSE = get_MSE(N, T, sims, DGP, model, crit)
            MSEs.append(
                {
                    'Method': model + ': ' + crit,
                    'MSE': MSE,
                }
            )

    MSE_df = pd.DataFrame(MSEs)
    sns.scatterplot(data=MSE_df, x='Method', y='MSE', marker="D", s=50)
    for i in range(len(MSE_df['Method'])):
        plt.annotate(round(MSE_df['MSE'][i], 3), (MSE_df['Method'][i], MSE_df['MSE'][i] + 0.0005),
                     ha='center')
    # plt.xlim((0, len(MSE_df['Method']) + 0.5))
    plt.ylim((-0.0005, max(MSE_df['MSE'] + 0.001)))
    plt.savefig(outpath + 'N' + str(N) + '_T' + str(T) + '_DGP' + str(DGP) + '_Sims' + str(sims) + '.png')
    plt.close()


for i in range(6):
    plot_MSE(100, 50, 100, i+1)

plot_MSE(100, 50, 10, 3)
