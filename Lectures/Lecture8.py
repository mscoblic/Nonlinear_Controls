# ME:5114 Nonlinear Control
# Lecture 8: Lyapunov
# Author: Prof. Shaoping Xiao, Mechanical Engineering, the University of Iowa

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spale
from scipy.integrate import odeint

print("-------------")
print("Example 1")
print("-------------")


def f1(X, t):
    x1, x2 = X
    return [x2, x1+(1+np.sin(x1)**2.0)*x2]

# initial condition
t0 = 0.0
x0 = [2.0, -2.0]

# scipy for solving IVP
t1=2.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f1, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-20, 5)
plt.xlim(t0, t1)
plt.plot(t, sol[:,0], 'k-', label='x1', linewidth =2)
plt.plot(t, sol[:,1], 'k--', label='x2', linewidth =2)
#plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1 or x2", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()




print("-------------")
print("Example 2 beta = 1")
print("-------------")

# Example 2

def f2(X, t):
    beta = 1.0
    x1, x2 = X
    return [x2, -9.81*np.sin(x1)-beta*x2]


   # plot several paths 

x1 = np.linspace(-3.0, 9.0, 30)
x2 = np.linspace(-5.0, 10.0, 20)
X1, X2 = np.meshgrid(x1, x2)
t = 0
u, v = np.zeros(X1.shape), np.zeros(X2.shape)
NI, NJ = X1.shape

for i in range(NI):
    for j in range(NJ):
        x = X1[i, j]
        y = X2[i, j]
        yprime = f2([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]
     

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x_1$', size=18)
plt.ylabel('$x_2$', size=18)
plt.xlim([-3, 9])
plt.ylim([-5, 10])
  
for x20 in [0, 2, 4, 6, 8]:
    tspan = np.linspace(0, 2, 200)
    x0 = [0.0, x20]
    xs = odeint(f2, x0, tspan)
    plt.plot(xs[:,0], xs[:,1], 'b-') # path
    plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
    plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

for x20 in [0,1, 4, 8]:
    tspan = np.linspace(0, 4, 200)
    x0 = [np.pi, x20]
    xs = odeint(f2, x0, tspan)
    plt.plot(xs[:,0], xs[:,1], 'b-') # path
    plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
    plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end
    
for x20 in [ 1, 3, 6]:
    tspan = np.linspace(0, 4, 200)
    x0 = [2*np.pi, x20]
    xs = odeint(f2, x0, tspan)
    plt.plot(xs[:,0], xs[:,1], 'b-') # path
    plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
    plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

plt.show()


# initial condition
t0 = 0.0
x0 = [2.0, -2.0]

# scipy for solving IVP
t1=5.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f2, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-8, 8)
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

# plot energy
plt.figure(figsize=(6,3))
plt.ylim(-10, 20)
plt.xlim(t0, t1)
plt.plot(t, 0.5*sol[:,1]**2.0 + 9.81*(1-np.cos(sol[:,0])), 'k-', label='energy', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("Energy", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# initial condition
t0 = 0.0
x0 = [4.0, -2.0]

# scipy for solving IVP
t1=5.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f2, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-8, 8)
plt.xlim(t0, t1)
plt.plot(t, sol[:,0], 'k-', label='x1', linewidth =2)
plt.plot(t, sol[:,1], 'k--', label='x2', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.plot(t, np.pi*2*np.ones(len(t)), 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1 or x2", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# plot energy
plt.figure(figsize=(6,3))
plt.ylim(-10, 30)
plt.xlim(t0, t1)
plt.plot(t, 0.5*sol[:,1]**2.0 + 9.81*(1-np.cos(sol[:,0]+np.pi)), 'k-', label='energy', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("Energy", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


print("-------------")
print("Example 2 beta = 0")
print("-------------")


# no friction
def f2(X, t):
    beta = 0.0
    x1, x2 = X
    return [x2, -9.81*np.sin(x1)-beta*x2]

# initial condition
t0 = 0.0
x0 = [2.0, -2.0]

# scipy for solving IVP
t1=5.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f2, x0, t)

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


# plot energy
plt.figure(figsize=(6,3))
plt.ylim(-10, 20)
plt.xlim(t0, t1)
plt.plot(t, 0.5*sol[:,1]**2.0 + 10*(1-np.cos(sol[:,0])), 'k-', label='energy', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("Energy", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

