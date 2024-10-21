
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt

# Training data (points and labels)
x = np.array([[0.2, 0.869], [0.687, 0.212], [0.822, 0.411], [0.738, 0.694],
              [0.176, 0.458], [0.306, 0.753], [0.936, 0.413], [0.215, 0.410],
              [0.612, 0.375], [0.784, 0.602], [0.612, 0.554], [0.357, 0.254],
              [0.204, 0.775], [0.512, 0.745], [0.498, 0.287], [0.251, 0.557],
              [0.502, 0.523], [0.119, 0.687], [0.495, 0.924], [0.612, 0.851]])

y = np.array([-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1])
y = y.astype('float').reshape(-1, 1)

# Parameters for the QP solver
C = 50.0
N = x.shape[0]

# Construct the matrices required for QP in standard form
H = (y @ y.T) * (x @ x.T)  # Quadratic term H = (y_i y_j) (x_i * x_j)
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((N, 1)))

# Equality constraints (A * lambda = b)
A = cvxopt_matrix(y.T)
b = cvxopt_matrix(np.zeros(1))

# Inequality constraints (0 <= lambda_i <= C)
G = cvxopt_matrix(np.vstack([-np.eye(N), np.eye(N)]))
h = cvxopt_matrix(np.hstack([np.zeros(N), np.ones(N) * C]))

# Solver settings
cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

# Solve the quadratic program
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
lamb = np.array(sol['x'])

# Compute weight vector w from lambdas
w = np.sum(lamb * y * x, axis=0)

# Identify support vectors
sv_idx = np.where(lamb > 1e-5)[0]
sv_x = x[sv_idx]
sv_y = y[sv_idx]

# Compute the bias term b using support vectors
b = np.mean(sv_y - np.dot(sv_x, w))

# Visualize the data points
plt.figure(figsize=(7, 7))
colors = ['red' if yi == 1 else 'blue' for yi in y]
plt.scatter(x[:, 0], x[:, 1], s=200, c=colors, alpha=0.7)
plt.xlim(0, 1)
plt.ylim(0, 1)

# Plot decision boundary
x1_dec = np.linspace(0, 1, 100)
x2_dec = -(w[0] * x1_dec + b) / w[1]
plt.plot(x1_dec, x2_dec, 'k-', lw=1.5, label='Decision Boundary')

# Plot positive and negative margins
w_norm = np.linalg.norm(w)
margin = 1 / w_norm
w_unit = w / w_norm
upper = -(w[0] * x1_dec + b - margin) / w[1]
lower = -(w[0] * x1_dec + b + margin) / w[1]
plt.plot(x1_dec, upper, 'k--', lw=1.0, label='Positive Margin')
plt.plot(x1_dec, lower, 'k--', lw=1.0, label='Negative Margin')

# Mark support vectors
plt.scatter(sv_x[:, 0], sv_x[:, 1], s=100, facecolors='none', edgecolors='k', marker='o')

# Compute and display slack variables
y_hat = np.dot(x, w) + b
slack = np.maximum(0, 1 - y.flatten() * y_hat)
for s, (x1, x2) in zip(slack, x):
    plt.annotate(str(s.round(2)), (x1 - 0.02, x2 + 0.03))

# Finalize plot
plt.legend()
plt.title(f'C = {C},  Σξ = {np.sum(slack).round(2)}')
plt.show()
