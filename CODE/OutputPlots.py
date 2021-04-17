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

    best_param, best_MSE, bestDOF = monte_carlo(params)
    MSE_avg = np.mean(best_MSE)
    return MSE_avg

def plot_MSE(N, T, sims, DGP):
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
    if DGP == 3:
        for i in range(len(MSE_df['Method'])):
            plt.annotate(round(MSE_df['MSE'][i], 5), (MSE_df['Method'][i], MSE_df['MSE'][i] + 0.00003),
                         ha='center')
        # plt.xlim((0, len(MSE_df['Method']) + 0.5))
        plt.ylim((-0.0001, max(MSE_df['MSE'] + 0.0002)))
    elif DGP == 2:
        for i in range(len(MSE_df['Method'])):
            plt.annotate(round(MSE_df['MSE'][i], 3), (MSE_df['Method'][i], MSE_df['MSE'][i] + 0.5),
                         ha='center')
        # plt.xlim((0, len(MSE_df['Method']) + 0.5))
        plt.ylim((-0.25, max(MSE_df['MSE'] + 1.1)))
    else:
        for i in range(len(MSE_df['Method'])):
            plt.annotate(round(MSE_df['MSE'][i], 3), (MSE_df['Method'][i], MSE_df['MSE'][i] + 0.05),
                         ha='center')
        # plt.xlim((0, len(MSE_df['Method']) + 0.5))
        plt.ylim((-0.05, max(MSE_df['MSE'] + 0.12)))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outpath + 'N' + str(N) + '_T' + str(T) + '_DGP' + str(DGP) + '_Sims' + str(sims) + '.png')
    plt.close()


for i in range(6):
    plot_MSE(100, 50, 10, i+1)

for i in range(6):
    plot_MSE(200, 500, 1000, i + 1)
