# ME:5114 Nonlinear Control
# Lecture 5: Nonlinear systems
# Author: Prof. Shaoping Xiao, Mechanical Engineering, the University of Iowa

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spale
from scipy.integrate import odeint

print("-------------")
print("Example 1")
print("-------------")

# Example 1
def f1(x, t):
    x1 = x
    return -x1+x1*x1

# initial displacement
t0 = 0.0

# scipy for solving IVP
t1=1.0
n= 1001 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at

x0 = 0.5
x1 = odeint(f1, x0, t)
x0 = 1.0
x2 = odeint(f1, x0, t)
x0 = 1.5
x3 = odeint(f1, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(0, 4)
plt.plot(t, x1, 'k-', label='x0=0.5', linewidth =2)
plt.plot(t, x2, 'k-.', label='x0=1.0', linewidth =2)
plt.plot(t, x3, 'k--', label='x0=1.5', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=12, frameon=False, labelspacing=0.5)
plt.ylabel("x(t)", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

print("-------------")
print("Example 5")
print("-------------")



# Example 5
def f5(x, t):
    x, y = X
    return [x*x, -y]

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
        yprime = f5([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]
     

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x$', size=15)
plt.ylabel('$y$', size=15)
plt.xlim([-5, 5])
plt.ylim([-4, 4])

tspan = np.linspace(0, 3, 200)
x0 = [-2, 2]
xs = odeint(f5, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

x0 = [-2, -2]
xs = odeint(f5, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

tspan = np.linspace(0, 0.5, 200)

x0 = [2, -2]
xs = odeint(f5, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

x0 = [2, 2]
xs = odeint(f5, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

plt.show()

print("-------------")
print("Example 6")
print("-------------")

# Example 6
def f6(X, t):
    x1, x2 = X
    return [x1*x1-1, x2*x2*x2-x2]

x1 = np.linspace(-2.0, 2.0, 15)
x2 = np.linspace(-2.0, 2.0, 15)
X1, X2 = np.meshgrid(x1, x2)
t = 0
u, v = np.zeros(X1.shape), np.zeros(X2.shape)
NI, NJ = X1.shape

for i in range(NI):
    for j in range(NJ):
        x = X1[i, j]
        y = X2[i, j]
        yprime = f6([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]
     

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x$', size=15)
plt.ylabel('$y$', size=15)
plt.xlim([-3, 3])
plt.ylim([-3, 3])

tspan = np.linspace(0, 0.5, 10)

x0 = [1.5, 1.5]
xs = odeint(f6, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

x0 = [0.8, 0.8]
xs = odeint(f6, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

x0 = [-2.8, 0.8]
xs = odeint(f6, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

x0 = [-2.8, 1.2]
xs = odeint(f6, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end


plt.show()


A = np.array([[-2, 0], [0, -1]])
le, x= spale.eig(A)   # solution from Scipy
print('the eigenvalues of matrix A are ', le)
print('the eigenvectors (columns) are \n', x)


print("-------------")
print("Example 7")
print("-------------")

# Example 7
def f7(X, t):
    x1, x2 = X
    return [x2, -9.81*np.sin(x1)]

x1 = np.linspace(-6.0, 6.0, 15)
x2 = np.linspace(-6.0, 6.0, 15)
X1, X2 = np.meshgrid(x1, x2)
t = 0
u, v = np.zeros(X1.shape), np.zeros(X2.shape)
NI, NJ = X1.shape

for i in range(NI):
    for j in range(NJ):
        x = X1[i, j]
        y = X2[i, j]
        yprime = f7([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]
     

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x_1$', size=18)
plt.ylabel('$x_2$', size=18)
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.show()

   # plot several paths 

x1 = np.linspace(-6.0, 15.0, 30)
x2 = np.linspace(-10.0, 10.0, 20)
X1, X2 = np.meshgrid(x1, x2)
t = 0
u, v = np.zeros(X1.shape), np.zeros(X2.shape)
NI, NJ = X1.shape

for i in range(NI):
    for j in range(NJ):
        x = X1[i, j]
        y = X2[i, j]
        yprime = f7([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]
     

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x_1$', size=18)
plt.ylabel('$x_2$', size=18)
plt.xlim([-5, 15])
plt.ylim([-10, 10])
  
for x20 in [0, 2, 4, 6, 8]:
    tspan = np.linspace(0, 2, 200)
    x0 = [0.0, x20]
    xs = odeint(f7, x0, tspan)
    plt.plot(xs[:,0], xs[:,1], 'b-') # path
    plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
    plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

for x20 in [0,1, 4, 8]:
    tspan = np.linspace(0, 4, 200)
    x0 = [np.pi, x20]
    xs = odeint(f7, x0, tspan)
    plt.plot(xs[:,0], xs[:,1], 'b-') # path
    plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
    plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end
    
for x20 in [ 1, 3, 6]:
    tspan = np.linspace(0, 4, 200)
    x0 = [2*np.pi, x20]
    xs = odeint(f7, x0, tspan)
    plt.plot(xs[:,0], xs[:,1], 'b-') # path
    plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
    plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

plt.show()

