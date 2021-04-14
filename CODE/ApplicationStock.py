from CODE.Methods import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import wrds


# ugly path because we have no dedicated data storage for now
df = pd.read_csv('/Users/jacobpichelmann/Dropbox/TSE Container/Y2/MachineLearning/datashare.csv')
# we have daily data on 30k stocks

# MAJOR ISSUE: data does not include returns. subseq. code just a MWE with simulated returns to see if method functions
# are easy to apply

# restrict time span to after 2015 and take a sample of stocks (don't need 30k) - maybe based on mvel?
df = df[df['DATE'] > 20150101]

sampled_firms = np.random.choice(df["permno"].unique(), 1000)
df_sampled = df.query('permno in @sampled_firms')

# add returns data
db = wrds.Connection()
db.list_tables(library='crsp')

db.describe_table(library='crsp', table='crsp_daily_data')

df_prc = db.get_table(library='crsp', table='monthly_returns')

db.close()


# check NAs
sns.heatmap(df_sampled.isnull(), cbar=False)

# because of many columns with missing NAs remove columns where more than 40% are missing
df = df_sampled[df_sampled.columns[df_sampled.isnull().mean() < 0.4]]


# transform predictor variables to array, removing all rows with NAs for now
df_clean = df_sampled[df_sampled.columns[~df_sampled.columns.isin(['permno', 'DATE', 'sic2'])]].dropna(axis=0)

# standardize data
scale = StandardScaler()
X = scale.fit_transform(df_clean)

# add random returns
Y = np.random.normal(0, 1, X.shape[0])


arrays = [Y, X]
keys = ['y', 'X']
data = {k: v for k, v in zip(keys, arrays)}

# add a test k
data['k'] = [4]

# works!
test = get_model_output(data, 'PC', 0)

# open question: are NAs in X a problem? for now we simply remove them





