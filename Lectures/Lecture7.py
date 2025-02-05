# ME:5114 Nonlinear Control
# Lecture 7: Solution existence and uniqueness
# Author: Prof. Shaoping Xiao, Mechanical Engineering, the University of Iowa

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spale
from scipy.integrate import odeint

print("-------------")
print("Example 1")
print("-------------")

# Example 1
def f1(x):
    return x*x*np.sin(1/x)

def df1(x):
    return 2*x*np.sin(1/x)-np.cos(1/x)

x=np.linspace(-2, 2.0, 100)
plt.figure(figsize=(6,3))
plt.ylim(-2, 2)
plt.plot(x, f1(x), 'k-', linewidth =2)
plt.plot(x, x*0, 'k--', label='y=0', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("f(x)", size=18)
plt.xlabel('x', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

x=np.linspace(-2, 2.0, 1000)
plt.figure(figsize=(6,3))
plt.ylim(-2, 2)
plt.plot(x, df1(x), 'k-', linewidth =2)
plt.plot(x, x*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=12, frameon=False, labelspacing=0.5)
plt.ylabel("f'(x)", size=18)
plt.xlabel('x', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

print("-------------")
print("Example 2")
print("-------------")


# Example 2
def f2(x):
    return 1/x


x=np.linspace(-2, 2.0, 100)
plt.figure(figsize=(6,3))
plt.ylim(-20, 20)
plt.plot(x, f2(x), 'k-', linewidth =2)
plt.plot(x, x*0, 'k--', label='y=0', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("f(x)", size=18)
plt.xlabel('x', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


print("-------------")
print("Example 3")
print("-------------")

# Example 3
def f3(x):
    return x**(1/3.0)

def df3(x):
    return 1/3.0*x**(-2/3.0)

x=np.linspace(0, 2.0, 100)
plt.figure(figsize=(6,3))
plt.ylim(-2, 2)
plt.plot(x, f3(x), 'k-', label='f(x)', linewidth =2)
plt.plot(x, x*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=12, frameon=False, labelspacing=0.5)
plt.ylabel("f(x)", size=15)
plt.xlabel('x', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

x=np.linspace(0, 2.0, 100)
plt.figure(figsize=(6,3))
plt.ylim(-2, 2)
plt.plot(x, df3(x), 'k-', label='df(x)', linewidth =2)
plt.plot(x, x*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=12, frameon=False, labelspacing=0.5)
plt.ylabel("f'(x)", size=15)
plt.xlabel('x', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


print("-------------")
print("Example 4")
print("-------------")
# Example 4
def f4(X, t):
    x1, x2 = X
    return [-x1+x1*x2, x2-x1*x2]

# scipy for solving IVP
t0=0
x0 = [4.0, 2.0]

t1=50.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f4, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-5, 5)
plt.plot(t, sol[:,0], 'k-', label='x1', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='x2', linewidth =2)
#plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

print("-------------")
print("Example 5")
print("-------------")
# Example 5
def f50(X, t):
    x1, x2 = X
    return [np.sqrt(x1*x1+5), x2-x1*x2]

# scipy for solving IVP
t0=0
x0 = [4.0, 0.0]

t1=5.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f50, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-50, 200)
plt.plot(t, sol[:,0], 'k-', label='x', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.xlabel('Time (s)', size=15)
plt.ylabel('x', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# Example 5
def f5(x):
    return np.sqrt(x*x+5)

def df5(x):
    return x/np.sqrt(x*x+5)

x=np.linspace(0, 20.0, 100)
plt.figure(figsize=(6,3))
plt.ylim(-2, 20)
plt.plot(x, f5(x), 'k-',  label='f(x)', linewidth =2)
plt.plot(x, x*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("f(x)", size=15)
plt.xlabel('x', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

x=np.linspace(0, 20.0, 1000)
plt.figure(figsize=(6,3))
plt.ylim(-1, 2)
plt.plot(x, df5(x), 'k-',  label='df(x)', linewidth =2)
plt.plot(x, x*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("f'(x)", size=15)
plt.xlabel('x', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

print("-------------")
print("Example 6")
print("-------------")
# Example 6
def f6(X, t):
    x1, x2 = X
    return [-x1*x1, x2-x1*x2]

# scipy for solving IVP
t0=0
x0 = [10.0, 0.0]

t1=1.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f6, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-5, 10)
plt.xlim(0, 1)
plt.plot(t, sol[:,0], 'k-', label='x (x0=10)', linewidth =2)
#plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

print("example 7 \n")
# Example 7
def f8(X, t):
    x1, x2 = X
    return [-x1*x1*x1, x2-x1*x2]

# scipy for solving IVP
t0=0
x0 = [10.0, 0.0]
x10 = [20.0, 0.0]

t1=0.1
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f8, x0, t)
sol1 = odeint(f8, x10, t)

plt.figure(figsize=(6,3))
plt.ylim(-5, 20)
plt.xlim(0, 0.1)
plt.plot(t, sol[:,0], 'k-', label='x (x0=10)', linewidth =2)
plt.plot(t, sol1[:,0], 'k--', label='y (y0=20)', linewidth =2)
#plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
