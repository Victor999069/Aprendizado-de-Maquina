import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

def compare_algorithms(X, max_clusters):
    results = []
    cluster_range = range(2, max_clusters +1)

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        clusters = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, clusters)
        results.append(('KMeans', n_clusters, silhouette_avg))

    
    for n_clusters in cluster_range:
        aggloremative = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = aggloremative.fit_predict(X)
        silhouette_avg = silhouette_score(X, clusters)
        results.append(('Aggloremative', n_clusters, silhouette_avg))

    for n_clusters in cluster_range:
        eps_value = np.arange(0.1, 0.9, 0.1)
        for eps in eps_value:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            clusters = dbscan.fit_predict(X)
            if len(set(clusters)) >1:
                silhouette_avg = silhouette_score(X, clusters)
                results.append(('DBSCAN', eps, silhouette_avg))

    return results

iris = datasets.load_iris()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(iris.data)

results = compare_algorithms(scaled_data, 10)
df = pd.DataFrame(results, columns=['Agrupador', 'Clusters', 'Score'])
print(df)