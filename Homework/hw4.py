import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

k = 2

def f1(X, t):
    x1, x2, x3 = X
    return [-k*x1, -x3**2-x2, x3*x2]

# initial conditions
t0 = 0.0
x0 = [5, 10, -7]

# scipy for solving IVP
t1=5.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f1, x0, t)

plt.figure(figsize=(8,7))
plt.ylim(-15, 15)
plt.xlim(t0, t1)
plt.plot(t, sol[:,0], 'r--', label='x1', linewidth=2)  # Red dashed line
plt.plot(t, sol[:,1], 'g--', label='x2', linewidth=2)  # Green dashed line
plt.plot(t, sol[:,2], 'b--', label='x3', linewidth=2)  # Blue dashed line
plt.axhline(y=0, color='k', linestyle='-', linewidth=1)  # Black solid line
#plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("State Variables", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

