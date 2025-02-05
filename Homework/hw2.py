# ME:5114 Nonlinear Control in Robotics Systems
# Homework 2
# Name: Mia Scoblic

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Problem 1 -------------
def f1(x,t):
    x1, x2 = x
    return [x2, np.cos(t)]

# Initial/Final Conditions
t0 = 0.0
tf = 5.0
x0 = [2, 0]

# Solve IVP
n = 100
dt = (tf-t0)/(n-1)
t = np.linspace(t0, tf, n) # points where solutions are solved at
sol = odeint(f1, x0, t)

plt.figure(figsize=(8,5))
plt.ylim(-2, 5)
plt.plot(t, sol[:,0], 'k-', color = 'red', label='State 1', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', color = 'blue', label='State 2', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("States", size=15)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Linear Model with u(t)=cos(t)", size=15)
plt.show()

# Problem 2 ------------
# Define the nonlinear system of ODEs
def f2(x, t):
    x1, x2 = x
    x1_dot = np.sin(x2)
    x2_dot = -x1**2 + np.cos(t)
    return [x1_dot, x2_dot]

# Initial/Final Conditions
t0 = 0.0
tf = 5.0
x0 = [2, 0]

# Solve IVP
n = 100
t = np.linspace(t0, tf, n)
sol = odeint(f2, x0, t)

# Plot results
plt.figure(figsize=(8,5))
plt.ylim(-8, 3)
plt.plot(t, sol[:,0], 'k-', color = 'red', label='State 1', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', color = 'blue', label='State 2', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend(loc='best', fontsize=15, frameon=False)
plt.ylabel("States", size=15)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Nonlinear Model with u(t)=cos(t)", size=15)
plt.show()