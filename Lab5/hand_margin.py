
from cvxopt import matrix as matrix
from cvxopt import solvers as solvers
import numpy as np
import matplotlib.pyplot as plt

# 3 data points
x = np.array([[1., 3.], [2., 2.], [1., 1.]])
y = np.array([[1.], [1.], [-1.]])

# ---- Calculate lambda using cvxopt ----

# Calculate H matrix (H = (y * x) @ (y * x).T)
H = (y @ y.T) * (x @ x.T)

# Construct the matrices required for QP in standard form
n = x.shape[0]
P = matrix(H)
q = matrix(-np.ones((n, 1)))
G = matrix(-np.eye(n))  # Negative identity matrix for λ >= 0 constraint
h = matrix(np.zeros(n))  # Zeros vector for Gλ <= h
A = matrix(y.reshape(1, -1), (1, n), 'd')
b = matrix(np.zeros(1))

# Solver parameters (optional but improve precision)
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10

# Perform QP to find λ
sol = solvers.qp(P, q, G, h, A, b)

# Solution of the QP, λ (Lagrange multipliers)
lamb = np.array(sol['x'])

# ---------------------------------------------------------------

# Calculate w using the lambda values (w = Σ λ_i * y_i * x_i)
w = np.sum(lamb * y * x, axis=0)

# Find support vectors (where λ > 1e-5)
sv_idx = np.where(lamb > 1e-5)[0]
sv_lamb = lamb[sv_idx]
sv_x = x[sv_idx]
sv_y = y[sv_idx]

# Calculate bias term b using support vectors (b = y_i - w^T * x_i)
b = np.mean(sv_y - np.dot(sv_x, w))

# Output the results
print('\nlambda =', np.round(lamb.flatten(), 3))
print('w =', np.round(w, 3))
print('b =', np.round(b, 3))

# Visualize the data points
plt.figure(figsize=(5, 5))
color = ['red' if a == 1 else 'blue' for a in y.flatten()]
plt.scatter(x[:, 0], x[:, 1], s=200, c=color, alpha=0.7)
plt.xlim(0, 4)
plt.ylim(0, 4)

# Visualize the decision boundary
x1_dec = np.linspace(0, 4, 100)
x2_dec = -(w[0] * x1_dec + b) / w[1]
plt.plot(x1_dec, x2_dec, c='black', lw=1.0, label='decision boundary')

# Visualize the positive & negative margins
w_norm = np.sqrt(np.sum(w ** 2))
w_unit = w / w_norm  # Unit vector in the direction of w
half_margin = 1 / w_norm

# Calculate positive and negative boundaries
upper = np.array([x1_dec, x2_dec + half_margin]).T
lower = np.array([x1_dec, x2_dec - half_margin]).T

plt.plot(upper[:, 0], upper[:, 1], '--', lw=1.0, label='positive boundary')
plt.plot(lower[:, 0], lower[:, 1], '--', lw=1.0, label='negative boundary')

# Highlight the support vectors
plt.scatter(sv_x[:, 0], sv_x[:, 1], s=50, marker='o', c='white')

# Annotate lambda values for each point
for s, (x1, x2) in zip(lamb, x):
    plt.annotate('λ=' + str(s[0].round(2)), (x1 - 0.05, x2 + 0.2))

plt.legend()
plt.show()

# Print the margin
print("\nMargin = {:.4f}".format(2 * half_margin))