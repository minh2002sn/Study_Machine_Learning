import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

B = np.zeros_like(A)

for i in range(3):
    A[:, i] = A[:, i]/(i + 1)

print(A)
