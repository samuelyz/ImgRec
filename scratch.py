import numpy as np
from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans

X = np.array([[1, 2],
              [1, 4],
              [1, 0],
              [4, 2],
              [4, 4],
              [4, 0]
             ])

kmeans = KMeans(n_clusters = 2, random_state=0).fit(X)
print("labels:", kmeans.labels_)
print("cluster centers:", kmeans.cluster_centers_)
print("inertia:", kmeans.inertia_)
print(kmeans.predict([[0, 0], [4, 4]]))

