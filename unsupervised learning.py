import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

cols = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "class"]
df = pd.read_csv("Data/seeds_dataset.txt", names=cols, sep=r"\s+", engine="python")




# for i in range(len(cols)-1):
#    for j in range(i+1,len(cols)-1):
#        x_label = cols[i]
#        y_label = cols[j]
#        sns.scatterplot(x = x_label, y = y_label, hue = cols[i], data = df)
#        plt.show()

x = "compactness"
y = "asymmetry"
X = df[[x, y]].values

kmeans = KMeans(n_clusters = 3).fit(X)
clusters = kmeans.labels_
#print(clusters)
cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=[x, y, "class"])

# K Means classes
sns.scatterplot(x=x, y=y, hue='class', data=cluster_df)
plt.plot()
#plt.show()
## old classes
sns.scatterplot(x=x, y=y, hue='class', data=df)
plt.plot()
#plt.show()

#higher dims
X = df[cols[:-1]].values

kmeans = KMeans(n_clusters = 3).fit(X)
cluster_df = pd.DataFrame(np.hstack((X, kmeans.labels_.reshape(-1, 1))), columns=df.columns)
# K Means classes
sns.scatterplot(x=x, y=y, hue='class', data=cluster_df)
plt.plot()
plt.show()