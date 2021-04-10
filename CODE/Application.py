from CODE.Methods import *

# ugly path because we have no dedicated data storage for now
df = pd.read_csv('/Users/jacobpichelmann/Dropbox/TSE Container/Y2/MachineLearning/datashare.csv')

# MAJOR ISSUE: data does not include returns. subseq. code just a MWE with simulated returns to see if method functions
# are easy to apply

# restrict time span to after 2005
df = df[df['DATE'] > 20050101]

# transform predictor variables to array
X = df[df.columns[~df.columns.isin(['permno', 'DATE', 'sic2'])]].to_numpy()

# add random returns
Y = np.random.normal(0, 1, len(df['DATE']))


arrays = [Y, X]
keys = ['y', 'X']
data = {k: v for k, v in zip(keys, arrays)}

# add a test k
data['k'] = [4]

# works!
test = get_model_output(data, 'PC', 0)

# open question: are NAs in X a problem?





