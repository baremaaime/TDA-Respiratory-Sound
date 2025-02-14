import numpy as np
from data_loader import data_loader
from feature_extraction import feature_extraction

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

signals, labels = data_loader()

results = []
for i, signal in enumerate(signals):
    print(i)
    result = feature_extraction(signal, embedding_dimension = 30, embedding_time_delay = 50, stride = 15)
    results.append(result)

X = np.array(results)
y = labels

rf = RandomForestClassifier(500)

cv_scores = cross_val_score(rf, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores))
print("Standard deviation:", np.std(cv_scores))