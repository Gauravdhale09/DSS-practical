# Use ‘pip install scikit-learn’ rather than ‘pip install sklearn’

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5]).reshape(-1,1)
y = 2 + 1.5 * x.squeeze() + np.random.normal(scale=0.5, size=len(x))
model = LinearRegression().fit(x, y)
a0 = model.intercept_
a1 = model.coef_[0]
print(f"Linear regression equation : y = {a0 : 2f} + {a1:2f} * x + epsilon")
plt.scatter(x, y, edgecolors='black', label='Training Data')
plt.plot(x, model.predict(x), color='blue', label='Linear Regression', linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression example')
plt.legend()
plt.show()


