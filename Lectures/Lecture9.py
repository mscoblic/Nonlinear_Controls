# ME:5114 Nonlinear Control
# Lecture 9: GAS
# Author: Prof. Shaoping Xiao, Mechanical Engineering, the University of Iowa

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spale
from scipy.integrate import odeint

print("-------------")
print("Example 2&4")
print("-------------")

#Example 4
def f4(X, t):
    x1, x2 = X
    return [x2, -0.5*x2*x2*x2-x1]

# initial condition
t0 = 0.0
x0 = [10.0, -2.0]

# scipy for solving IVP
t1=50.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f4, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-12, 12)
plt.xlim(t0, t1)
plt.plot(t, sol[:,0], 'k-', label='x1', linewidth =2)
plt.plot(t, sol[:,1], 'k--', label='x2', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1 or x2", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

plt.figure(figsize=(6,3))
plt.ylim(-10,100)
plt.xlim(t0, t1)
plt.plot(t, sol[:,0]**2.0 +sol[:,1]**2.0, 'k-', label='V', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("V", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Example 4
x = np.linspace(-5.0, 5.0, n) # points where solutions are solved at
plt.plot(x, np.cosh(x), 'k-', label='cosh(x)', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("cosh(x)", size=18)
plt.xlabel('x', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


x = np.linspace(-5.0, 5.0, n) # points where solutions are solved at
plt.plot(x, np.tanh(x), 'k--', label='tanh(x)', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("cosh(x)", size=18)
plt.xlabel('x', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


print("-------------")
print("Example 5")
print("-------------")


#Example 5
def f5(X, t):
    x1, x2 = X
    return [x2-x1*(x1*x1+x2*x2), -x1-x2*(x1*x1+x2*x2)]

# initial condition
t0 = 0.0
x0 = [10.0, -2.0]

# scipy for solving IVP
t1=5.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f5, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-12, 12)
plt.xlim(t0, t1)
plt.plot(t, sol[:,0], 'k-', label='x1', linewidth =2)
plt.plot(t, sol[:,1], 'k--', label='x2', linewidth =2)
#plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1 or x2", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

plt.ylim(-10,50)
plt.xlim(t0, t1)
plt.plot(t, 0.5*sol[:,0]**2.0 +0.5*sol[:,1]**2.0, 'k-', label='V', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("V", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

x1 = np.linspace(-100.0, 100.0, 15)
x2 = np.linspace(-100.0, 100.0, 15)
X1, X2 = np.meshgrid(x1, x2)
t = 0
u, v = np.zeros(X1.shape), np.zeros(X2.shape)
NI, NJ = X1.shape

for i in range(NI):
    for j in range(NJ):
        x = X1[i, j]
        y = X2[i, j]
        yprime = f5([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]

plt.figure(figsize=(6,6))

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x_1$', size=15)
plt.ylabel('$x_2$', size=15)
plt.xlim([-100, 100])
plt.ylim([-100, 100])
plt.show()