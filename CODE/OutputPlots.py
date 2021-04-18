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
    'method': eval}

    best_param, best_MSE, bestDOF, cv_tim = monte_carlo(params)
    MSE_avg = np.mean(best_MSE)
    return MSE_avg

def plot_MSE(N, T, sims, DGP, cor):
    MSEs = []

    for model in ['PC', 'Ridge', 'PLS', 'LF']:
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
    plt.ylim((-0.0005, 1.1 * max(MSE_df['MSE'])))
    size = 1.1 * max(MSE_df['MSE']) + 0.0005
    for i in range(len(MSE_df['Method'])):
        plt.annotate(round(MSE_df['MSE'][i], 3), (MSE_df['Method'][i], cor * size + MSE_df['MSE'][i]),
                     ha='center')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outpath + 'N' + str(N) + '_T' + str(T) + '_DGP' + str(DGP) + '_Sims' + str(sims) + '.png')
    plt.close()


for i in range(6):
    plot_MSE(100, 50, 1000, i+1, cor=0.05)

plot_MSE(100, 50, 1000, 3, cor=0.05)


for i in range(6):
    plot_MSE(200, 500, 25, i + 1, cor=0.05)

plot_MSE(200, 500, 25, 4, cor=0.05)

