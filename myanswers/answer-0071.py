import numpy as np
from sklearn.cluster import FeatureAgglomeration

def comprimir_sensores_correlacionados(X, n_clusters):
    modelo = FeatureAgglomeration(n_clusters=n_clusters)
    modelo.fit(X)
    return modelo.transform(X)
