# eqn is: y = a0 + a1x1 + a2x2 + .... + anxn + E

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([[1, 2], [2, 3], [4, 5], [5, 6]])
y = 2 + 1.5 * x[:,0] + 0.8 * x[:,1] + np.random.normal(scale=0.5, size=len(x))
model = LinearRegression().fit(x, y)
a0 = model.intercept_
a1, a2 = model.coef_
print(f"Multilinear regression eqn : y={a0: .2f} + {a1: .2f} * x1 + {a2: .2f} * x2 + epsilon")
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x[:, 0], x[:, 1], y, color='black', label='Training Data')
ax.scatter(x[:, 0], x[:, 1], model.predict(x), color='red', label='predictions')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Multilinear Regression Example')
plt.legend()
plt.show()

