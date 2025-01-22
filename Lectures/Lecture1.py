# ME:5114 Nonlinear Control in Robotics Systems
# Lecture 1: Introduction
# Author: Prof. Shaoping Xiao, Mechanical Engineering, the University of Iowa

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

print("\n")
print("------------Example 4----------")
print("\n")
# Example 4

def f4(x, t):  # RHS of ODE dx = f(x,t)
    return -2*x+t

def g(t):  # analytical solution
    return 5.25*np.exp(-2*t)+t/2.0-1/4.0

# initial condition
t0 = 0.0
x0 = [5.0]

# scipy for solving IVP
t1=5.0
n=15 # number of points
t = np.linspace(t0, t1, n) # points where the solutions are solved at
sol = odeint(f4, x0, t) # x= sol[:,0]


plt.figure(figsize=(6,3))
plt.ylim(-2, 5)
plt.plot(t, sol[:,0], 'x', label='ODE solver', linewidth =2)
plt.plot(t, g(t), 'k-', label='Exact solution', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x", size=15)
plt.xlabel('Time', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


print("\n")
print("------------Example 6----------")
print("\n")
# Example 6

def f6(y, t):
    x, v = y
    return [v, 9.81-0.25/68.1*v*v]

# initial condition
t0 = 0.0
y0 = [0.0, 0.0]

# scipy for solving IVP
t1=2.0
n=11 # number of points
t = np.linspace(t0, t1, n) # points where the solutions are solved at
sol = odeint(f6, y0, t)

# list solutions
for i in range (0, n):
    print('at t = {:.2f}, x = {:.3f} and v = {:.3f}'.format(t0+(t1-t0)/(n-1)*i, sol[i,0], sol[i,1]))


print("\n")
print("------------Example 7----------")
print("\n")
# Example 7

def f7(x, t):
    x1, x2 = x
    return [x2, -x2+12*x1]

def g(t):
    return 16/7.0*np.exp(3*t)+5/7.0*np.exp(-4*t)

t0 = 0.0
x0 = [3.0, 4.0]

# scipy for solving IVP
t1=1.0
n=15 # number of points
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f7, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(0, 80)
plt.plot(t, sol[:,0], 'x', label='ODE solver', linewidth =2)
plt.plot(t, g(t), 'k-', label='Exact solution', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x", size=15)
plt.xlabel('Time', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
