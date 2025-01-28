# ME:5114 Nonlinear Control in Robotics Systems
# Homework 1
# Name: Mia Scoblic

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

'''
# Problem 1

# RHS
def f1(x, t):
    return np.cos(2*t)

# Analytical Solution
def g1(x):
    return

# Initial Conditions
t0 = 0.0
x0 = [2]

# Solving IVP
t1 = 5.0
n = 15

t = np.linspace(t0, t1, n)
sol = odeint(f1, x0, t)

# Plot
plt.figure(figsize=(6,3))
plt.ylim(-2, 5)
plt.plot(t, sol[:,0], 'x', label='ODE solver', linewidth =2)
plt.plot(t, g1(t), 'k-', label='Exact solution', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x", size=15)
plt.xlabel('Time', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
'''

# Problem 2

# RHS
def f2(x, t):
    x1, x2 = x
    return [x2, 4*x2-4*x1]

# Analytical Solution
def g2(t):
    return (2-4*t)*np.exp(2*t)

# Initial Conditions
t0 = 0
x0 = [2, 0]

# Solving IVP
t1 = 0.5
n = 15 # number of points
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f2, x0, t)

# Plot
plt.figure(figsize=(6,3))
plt.ylim(0, 2.5)
plt.plot(t, sol[:,0], 'x', label='ODE solver', linewidth =2)
plt.plot(t, g2(t), 'k-', label='Exact solution', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x", size=15)
plt.xlabel('Time', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

