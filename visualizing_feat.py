import numpy as np
import pandas as pd
import pylab
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Complete_Features_1.csv').dropna().reset_index(drop = True)
data = df.iloc[: , 1:]


print(data.head())
cols = data.columns[1:25]
# plt.figure(figsize=(13,13))
# sns.set(font_scale=1.2)
# sns.pairplot(data[cols], height=2.0)
# plt.show()
print(data['Ratio STDs'])
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_std = stdsc.fit_transform(data[cols].iloc[:,range(0,24)].values)

cov_mat = np.cov(X_std.T, bias= True)

import seaborn as sns
plt.figure(figsize=(11,11))
# sns.set(font_scale=1.2)
hm = sns.heatmap(cov_mat,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 7},
                 yticklabels=cols,
                 xticklabels=cols)
plt.title('Covariance matrix showing correlation coefficients')
plt.tight_layout()
plt.show()