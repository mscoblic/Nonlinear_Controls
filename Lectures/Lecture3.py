# ME:5114 Nonlinear Control
# Lecture 3: Limitations of linear systems
# Author: Prof. Shaoping Xiao, Mechanical Engineering, the University of Iowa

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spale
from scipy.integrate import odeint


print("\n")
print("------------Example 6----------")
print("\n")

# Example 6: linear model
def f1(x, t):
    x1 = x
    return -x1

# initial displacement
t0 = 0.0
x0 = 2

# scipy for solving IVP
t1=5.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f1, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-1, 2)
plt.plot(t, sol, 'k-', label='x', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x", size=15)
plt.xlabel('Time', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Example 6: another linear model
def f12(x, t):
    x1 = x
    return x1-1

# initial displacement
t0 = 0.0
x0 = 2

# scipy for solving IVP
t1=5.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f12, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-1, 15)
plt.plot(t, sol, 'k-', label='x', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x", size=15)
plt.xlabel('Time', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Example 2: nonlinear model
def fx(x0,t):
    return x0*np.exp(-t)/(1-x0+x0*np.exp(-t))
# initial displacement
t0 = 0.0

# scipy for solving IVP
t1=5.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol0 = fx(0.9, t)
sol1 = fx(1, t)
sol2 = fx(1.1, t)

plt.figure(figsize=(6,3))
plt.ylim(-1, 4)
plt.xlim(0, 5)
plt.plot(t, sol0, 'k-', label='x0=0.9', linewidth =2)
plt.plot(t, sol1, 'k--', label='x0=1', linewidth =2)
plt.plot(t, sol2, 'k-.', label='x0=1.1', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x", size=15)
plt.xlabel('Time', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

print("-------------")
print("Example 7")
print("-------------")


# Example 3: 
def f2(x, t):
    x1, x2 = x
    u = np.sin(t)
    return [x1+5*x2-2*u, 8*x1+4*x2+2*u]

# initial displacement
t0 = 0.0
x0 = [-1, 1]

# scipy for solving IVP
t1=5.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f2, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-2, 2)
plt.plot(t, sol[:,0], 'k-', label="x1", linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label="x2", linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1", size=15)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title ("x0=[-1, 1], u=sin(t)", size=15)
plt.show()

Aa = np.array([[1,5],[8,4]])
print("eigenvalues are", np.linalg.eig(Aa))

print("-------------")
print("Example 8")
print("-------------")


def f4(x, t):
    x1, x2 = x
    return [x2, 3*x2-np.sin(t)]

# initial displacement
t0 = 0.0
x0 = [1, 0]

# scipy for solving IVP
t1=2.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f4, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-2, 2)
plt.plot(t, sol[:,0], 'k-', label="x1", linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label="x2", linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x", size=15)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title ("x0=[1, 0], u=sin(t)", size=15)
plt.show()


print("-------------")
print("Example 9")
print("-------------")


def f5(x, t):
    x1, x2 = x
    return [-x1, 3*x2-np.sin(t)]

# initial displacement
t0 = 0.0
x0 = [1, 0]

# scipy for solving IVP
t1=2.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f5, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-2, 2)
plt.plot(t, sol[:,0], 'k-', label="x1", linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label="x2", linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x", size=15)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title ("x0=[1, 0], u=sin(t)", size=15)
plt.show()

print("-------------")
print("Example 10")
print("-------------")


def f6(x, t):
    x1, x2 = x
    return [x1, 3*x2-np.sin(t)]

# initial displacement
t0 = 0.0
x0 = [1, 0]

# scipy for solving IVP
t1=2.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f6, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-2, 5)
plt.plot(t, sol[:,0], 'k-', label="x1", linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label="x2", linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x", size=15)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title ("x0=[1, 0], u=sin(t)", size=15)
plt.show()

