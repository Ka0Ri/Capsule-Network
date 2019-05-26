import numpy as np
import matplotlib.pyplot as plt
def squash(v):
    s = np.linalg.norm(v)
    return (s**2/(1+ s**2))*(v/s)

def softmax(b):
    c = np.exp(b)/np.sum(np.exp(b))
    return c

k = 10
u = np.random.rand(k, 2)*2 - 1
u = squash(u)
b = np.zeros(k)
n = 1000
result = []
for i in range(n):
    c = softmax(b)
    v = np.dot(np.reshape(c, (1, k)), u) 
    # v = squash(v)
    b = b + np.dot(v, u.T)
    if(i % 10 == 0):
        result.append(v[0])

# for v in result:
#     origin = [0], [0] # origin point
#     plt.quiver(*origin, u[:,0], u[:,1], color='b', scale=1)
#     plt.quiver(*origin, v[0], v[1], color='r', scale=1)
#     plt.show()

origin = [0], [0] # origin point
plt.quiver(*origin, u[:,0], u[:,1], color='b', scale=1)
plt.quiver(*origin, result[-1][0], result[-1][1], color='r', scale=1)
plt.show()