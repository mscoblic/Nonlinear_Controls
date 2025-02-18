import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spale
from scipy.integrate import odeint

# Linearized Phase Portrait

"""
def f1(X, t):
    x, y = X
    return [-x+((x*x)*y), -y]


x1 = np.linspace(-5.0, 5.0, 20)
x2 = np.linspace(-3.0, 3.0, 20)
X1, X2 = np.meshgrid(x1, x2)
t = 0
u, v = np.zeros(X1.shape), np.zeros(X2.shape)
NI, NJ = X1.shape

for i in range(NI):
    for j in range(NJ):
        x = X1[i, j]
        y = X2[i, j]
        yprime = f1([x, y], t)
        u[i, j] = yprime[0]
        v[i, j] = yprime[1]

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x$', size=15)
plt.ylabel('$y$', size=15)
plt.xlim([-5, 5])
plt.ylim([-3, 3])

# Linearized trajectories using multiple initial conditions

tspan = np.linspace(0, 3, 300)

# Stable
#'''
x0 = [4, 2]
xs = odeint(f1, x0, tspan)
plt.plot(xs[:, 0], xs[:, 1], 'b-')  # path
plt.plot([xs[0, 0]], [xs[0, 1]], 'o')  # start
plt.plot([xs[-1, 0]], [xs[-1, 1]], 's')  # end
#'''

# Stable
#'''
x0 = [0, 0]
xs = odeint(f1, x0, tspan)
plt.plot(xs[:, 0], xs[:, 1], 'b-')  # path
plt.plot([xs[0, 0]], [xs[0, 1]], 'o')  # start
plt.plot([xs[-1, 0]], [xs[-1, 1]], 's')  # end
#'''

# Unstable?
#'''
x0 = [-2.0, -1.0]
xs = odeint(f1, x0, tspan)
plt.plot(xs[:, 0], xs[:, 1], 'b-')  # path
plt.plot([xs[0, 0]], [xs[0, 1]], 'o')  # start
plt.plot([xs[-1, 0]], [xs[-1, 1]], 's')  # end
#'''

# Stable
#'''
x0 = [-2, -2]
xs = odeint(f1, x0, tspan)
plt.plot(xs[:, 0], xs[:, 1], 'b-')  # path
plt.plot([xs[0, 0]], [xs[0, 1]], 'o')  # start
plt.plot([xs[-1, 0]], [xs[-1, 1]], 's')  # end
#'''

#'''
x0 = [-2.0, -3.0]
xs = odeint(f1, x0, tspan)
plt.plot(xs[:, 0], xs[:, 1], 'b-')  # path
plt.plot([xs[0, 0]], [xs[0, 1]], 'o')  # start
plt.plot([xs[-1, 0]], [xs[-1, 1]], 's')  # end
#'''

plt.show()

"""

# Nonlinear Phase Portrait

def f2(x, t):
    x1, x2 = x
    x1_dot = -x1+(x1**2*x2)
    x2_dot = -x2
    return [x1_dot, x2_dot]

x1 = np.linspace(-5.0, 5.0, 20)
x2 = np.linspace(-3.0, 3.0, 20)
X1, X2 = np.meshgrid(x1, x2)
t = 0
u, v = np.zeros(X1.shape), np.zeros(X2.shape)
NI, NJ = X1.shape

for i in range(NI):
    for j in range(NJ):
        x = X1[i, j]
        y = X2[i, j]
        yprime = f2([x, y], t)
        u[i, j] = yprime[0]
        v[i, j] = yprime[1]

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x$', size=15)
plt.ylabel('$y$', size=15)
plt.xlim([-5, 5])
plt.ylim([-3, 3])

# Linearized trajectories using multiple initial conditions

tspan = np.linspace(0, 3, 300)

# Stable
#'''
x0 = [4, 2]
xs = odeint(f2, x0, tspan)
plt.plot(xs[:, 0], xs[:, 1], 'b-')  # path
plt.plot([xs[0, 0]], [xs[0, 1]], 'o')  # start
plt.plot([xs[-1, 0]], [xs[-1, 1]], 's')  # end
#'''

# Stable
#'''
x0 = [0, 0]
xs = odeint(f2, x0, tspan)
plt.plot(xs[:, 0], xs[:, 1], 'b-')  # path
plt.plot([xs[0, 0]], [xs[0, 1]], 'o')  # start
plt.plot([xs[-1, 0]], [xs[-1, 1]], 's')  # end
#'''

# Unstable?
#'''
x0 = [-2.0, -1.0]
xs = odeint(f2, x0, tspan)
plt.plot(xs[:, 0], xs[:, 1], 'b-')  # path
plt.plot([xs[0, 0]], [xs[0, 1]], 'o')  # start
plt.plot([xs[-1, 0]], [xs[-1, 1]], 's')  # end
#'''

# Stable
#'''
x0 = [-2, -2]
xs = odeint(f2, x0, tspan)
plt.plot(xs[:, 0], xs[:, 1], 'b-')  # path
plt.plot([xs[0, 0]], [xs[0, 1]], 'o')  # start
plt.plot([xs[-1, 0]], [xs[-1, 1]], 's')  # end
#'''

#'''
x0 = [-2.0, -3.0]
xs = odeint(f2, x0, tspan)
plt.plot(xs[:, 0], xs[:, 1], 'b-')  # path
plt.plot([xs[0, 0]], [xs[0, 1]], 'o')  # start
plt.plot([xs[-1, 0]], [xs[-1, 1]], 's')  # end
#'''

plt.show()
