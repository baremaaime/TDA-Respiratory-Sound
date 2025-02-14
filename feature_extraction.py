import numpy as np

from gtda.time_series import TakensEmbedding
from gtda.time_series import SingleTakensEmbedding
from ripser import ripser
from gtda.diagrams import PersistenceEntropy, NumberOfPoints, Amplitude
from teaspoon.ML import feature_functions as Ff

from sklearn.pipeline import make_union

def feature_extraction(signal, embedding_dimension = 30, embedding_time_delay = 300, stride = 10):
    embedder = SingleTakensEmbedding(
        parameters_type="search", n_jobs=4, time_delay=embedding_time_delay, dimension=embedding_dimension, stride=stride
    )
    
    y_noise_embedded = fit_embedder(embedder, signal)

    res = ripser(y_noise_embedded, n_perm=100)
    dgms_sub = res['dgms']

    res = convert_dgm(dgms_sub)
    
    persistence_entropy = PersistenceEntropy()

    # calculate topological feature matrix
    X_pe = persistence_entropy.fit_transform(res[None,:,:])

    test = dgms_sub[0][:-1]

    test_1 = dgms_sub[1]

    # compute feature matrix
    FN = 5
    FeatureMatrix, _, _ = Ff.F_CCoordinates(test[None,:,:], FN)

    X_cc_0 = FeatureMatrix[-8]

    FeatureMatrix, _, _ = Ff.F_CCoordinates(test_1[None,:,:], FN)

    X_cc_1 = FeatureMatrix[-9]

    # Listing all metrics we want to use to extract diagram amplitudes
    metrics = [
        {"metric": "bottleneck", "metric_params": {}},
        {"metric": "wasserstein", "metric_params": {"p": 2}},
        {"metric": "betti", "metric_params": {"p": 2, "n_bins": 100}},
        {"metric": "heat", "metric_params": {"p": 2, "sigma": 1.6, "n_bins": 100}},
        {"metric": "heat", "metric_params": {"p": 2, "sigma": 3.2, "n_bins": 100}},
    ]

    feature_union = make_union(
    PersistenceEntropy(normalize=True),
    NumberOfPoints(n_jobs=-1),
    *[Amplitude(**metric, n_jobs=-1) for metric in metrics]
    )

    single_data = feature_union.fit_transform(res[None,:,:])

    X_metrics = single_data

    single_X_train = np.concatenate((X_cc_0,X_cc_1, X_metrics), axis=None)

    return single_X_train
    
def convert_dgm(dgm):
    Arr = dgm.copy()
    Arr[0] = Arr[0][:-1]
    col_a  = np.zeros(Arr[0].shape[0])
    Arr[0] = np.column_stack((Arr[0], col_a))
    
    col_b  = np.ones(Arr[1].shape[0], dtype=int)
    Arr[1] = np.column_stack((Arr[1], col_b))
    temp_1 = list(Arr[0])
    temp_2 = list(Arr[1])
    temp_1.extend(temp_2)
    return np.asarray(temp_1)

def fit_embedder(embedder, y, verbose=True):
    y_embedded = embedder.fit_transform(y)

    if verbose:
        print(f"Shape of embedded time series: {y_embedded.shape}")
        print(f"Optimal embedding dimension is {embedder.dimension_} and time delay is {embedder.time_delay_}")

    return y_embedded

if __name__ == "__main__":
    feature_extraction()