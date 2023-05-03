import numpy as np

def reLU_grad(a):
    """reLU_grad
    TODO:
    Compute gradient of ReLU with respect to input
    :param x: output of ReLU
    """
    #[TODO 1.1]
    grad = a
    grad[grad<0] = 0
    grad[grad>0] = 1
    return grad

a = np.array([[-2, -1, 0, 1, -5], [-2, -1, 0, 1, -5]])
b = np.array([[1, 2], [2, 1], [3, 5], [7, 9], [0, 1]])
c = np.array([-2, -1, 0, 1, 5])
print(b.shape)
print(b[0::3])
print(c.shape)
c = c.reshape(5, 1)
print(c.shape)
# print(np.square(a))
# print(np.mean(b))
# print(np.exp(a).sum(axis=1, keepdims=True))
# print(a.sum(axis=1, keepdims=True))
# print(np.max(a, axis=1, keepdims=True))
