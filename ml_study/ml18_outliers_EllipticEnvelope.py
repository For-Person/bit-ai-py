import numpy as np

oliers = np.array([-50, -10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])
oliers = oliers.reshape(-1, 1)
print(oliers.shape) # (14, 1)

from sklearn.covariance import EllipticEnvelope # 이상치 탐지
outlers = EllipticEnvelope(contamination=.1)
outlers.fit(oliers)
result = outlers.predict(oliers)

print(result) # [-1  1  1  1  1  1  1  1  1  1  1  1  1 -1]
print(result.shape) # (14,)