# ME:5114 Nonlinear Control in Robotics Systems
# Lecture 2: Linear Systems 
# Author: Prof. Shaoping Xiao, Mechanical Engineering, the University of Iowa

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spale
from scipy.integrate import odeint

print("\n")
print("------------Example 3----------")
print("\n")
# Example 3


# Example 3  - no input
def f3(x, t):
    x1, x2 = x
    return [x2, -x2-x1]

# initial displacement
t0 = 0.0
x0 = [10.0, 0.0]

# scipy for solving IVP
t1=10.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f3, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-10, 10)
plt.plot(t, sol[:,0], 'k-', label='displacement', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='velocity', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1 or x2", size=15)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("no input; x1(0)=10, x2(0)=0", size=15)
plt.show()


# Example 3 - initial velocity
def f4(x, t):
    x1, x2 = x
    return [x2, -x2-x1]

# initial displacement
t0 = 0.0
x0 = [0.0, 10.0]

# scipy for solving IVP
t1=10.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f4, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-10, 10)
plt.plot(t, sol[:,0], 'k-', label='displacement', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='velocity', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1 or x2", size=15)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("no input; x1(0)=0, x2(0)=10", size=15)
plt.show()


# Example 3 with sine input
def f5(x, t):
    x1, x2 = x
    return [x2, np.sin(t)-x2-x1]

# initial condition
t0 = 0.0
x0 = [0.0, 0.0]

# scipy for solving IVP
t1=20.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f5, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-2, 3)
plt.plot(t, sol[:,0], 'k-', label='displacement', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='velocity', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1 or x2", size=15)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("input = sin(t)", size=15)
plt.show()



print("\n")
print("------------Example 4----------")
print("\n")

# Example 2: pendulum - linearization at theta=0
def f2(x, t):
    x1, x2 = x
    return [x2, -9.81*x1]

# initial displacement
t0 = 0.0
x0 = [.1, 0.0]

# scipy for solving IVP
t1=10.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f2, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-0.5, 1)
plt.plot(t, sol[:,0], 'k-', label='angle', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='angular velocity', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1 or x2", size=15)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("around angle=0", size=15)
plt.show()


# Example 2: pendulum - linearization at theta=pi
def f21(x, t):
    x1, x2 = x
    return [x2, 9.81*x1]

# initial displacement
t0 = 0.0
x0 = [.1, 0.0]

# scipy for solving IVP
t1=10.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f21, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-10, 100)
plt.plot(t, sol[:,0], 'k-', label='angle', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='angular velocity', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1 or x2", size=15)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("around angle=PI", size=15)
plt.show()
