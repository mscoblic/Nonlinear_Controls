# ME:5114 Nonlinear Control
# Lecture 6: Nonlinear systems, stability, limit cycles and bifurcation
# Author: Prof. Shaoping Xiao, Mechanical Engineering, the University of Iowa

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spale
from scipy.integrate import odeint

print("-------------")
print("Example 1")
print("-------------")

# Example 1
def f1(X, t):
    x1, x2 = X
    return [x2, -9.81*np.sin(x1)-x2]

x1 = np.linspace(-8.0, 10.0, 15)
x2 = np.linspace(-8.0, 10.0, 15)
X1, X2 = np.meshgrid(x1, x2)
t = 0
u, v = np.zeros(X1.shape), np.zeros(X2.shape)
NI, NJ = X1.shape

for i in range(NI):
    for j in range(NJ):
        x = X1[i, j]
        y = X2[i, j]
        yprime = f1([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]

plt.figure(figsize=(6,6))

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x_1$', size=15)
plt.ylabel('$x_2$', size=15)
plt.xlim([-8, 10])
plt.ylim([-8, 10])
plt.show()

     
plt.figure(figsize=(8,8))

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x_1$', size=15)
plt.ylabel('$x_2$', size=15)
plt.xlim([-8, 20])
plt.ylim([-8, 20])

for x20 in [4, 8, 12, 16, 20]:
    tspan = np.linspace(0, 5, 200)
    x0 = [0.0, x20]
    xs = odeint(f1, x0, tspan)
    plt.plot(xs[:,0], xs[:,1], 'b-') # path
    plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
    plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

for x20 in [-4, -8]:
    tspan = np.linspace(0, 5, 200)
    x0 = [np.pi, x20]
    xs = odeint(f1, x0, tspan)
    plt.plot(xs[:,0], xs[:,1], 'b-') # path
    plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
    plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

for x20 in [-4, -8]:
    tspan = np.linspace(0, 5, 200)
    x0 = [2*np.pi, x20]
    xs = odeint(f1, x0, tspan)
    plt.plot(xs[:,0], xs[:,1], 'b-') # path
    plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
    plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end

plt.show()


print("-------------")
print("Example 2")
print("-------------")

# Example 2
A = np.array([[0, 1], [-9.81, -1]])
le, x= spale.eig(A)   # solution from Scipy
print('the eigenvalues of matrix A are ', le)
print('the eigenvectors (columns) are \n', x)

A = np.array([[0, 1], [9.81, -1]])
le, x= spale.eig(A)   # solution from Scipy
print('the eigenvalues of matrix A are ', le)
print('the eigenvectors (columns) are \n', x)


print("-------------")
print("Example 3")
print("-------------")

# Example 3 - linearization
def f3(X, t):
    x1, x2 = X
    return [-x2, x1]

A = np.array([[0, 1], [-1, 0]])
le, x= spale.eig(A)   # solution from Scipy
print('the eigenvalues of matrix A are ', le)
print('the eigenvectors (columns) are \n', x)

x1 = np.linspace(-8.0, 10.0, 15)
x2 = np.linspace(-8.0, 10.0, 15)
X1, X2 = np.meshgrid(x1, x2)
t = 0
u, v = np.zeros(X1.shape), np.zeros(X2.shape)
NI, NJ = X1.shape

for i in range(NI):
    for j in range(NJ):
        x = X1[i, j]
        y = X2[i, j]
        yprime = f3([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]

plt.figure(figsize=(6,6))

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x_1$', size=15)
plt.ylabel('$x_2$', size=15)
plt.xlim([-8, 10])
plt.ylim([-8, 10])
plt.show()


# Example 3 - nonlinear

mu = 1.0
def f3p(X, t):
    x1, x2 = X
    return [-x2-mu*x1*(x1*x1+x2*x2), x1-mu*x2*(x1*x1+x2*x2)]

x1 = np.linspace(-8.0, 10.0, 15)
x2 = np.linspace(-8.0, 10.0, 15)
X1, X2 = np.meshgrid(x1, x2)
t = 0
u, v = np.zeros(X1.shape), np.zeros(X2.shape)
NI, NJ = X1.shape

for i in range(NI):
    for j in range(NJ):
        x = X1[i, j]
        y = X2[i, j]
        yprime = f3p([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]

plt.figure(figsize=(6,6))

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x_1$', size=15)
plt.ylabel('$x_2$', size=15)
plt.xlim([-8, 10])
plt.ylim([-8, 10])
plt.title("mu=1.0", size = 15)
plt.show()

mu = -1.0
def f3n(X, t):
    x1, x2 = X
    return [-x2-mu*x1*(x1*x1+x2*x2), x1-mu*x2*(x1*x1+x2*x2)]

x1 = np.linspace(-8.0, 10.0, 15)
x2 = np.linspace(-8.0, 10.0, 15)
X1, X2 = np.meshgrid(x1, x2)
t = 0
u, v = np.zeros(X1.shape), np.zeros(X2.shape)
NI, NJ = X1.shape

for i in range(NI):
    for j in range(NJ):
        x = X1[i, j]
        y = X2[i, j]
        yprime = f3n([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]

plt.figure(figsize=(6,6))

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x_1$', size=15)
plt.ylabel('$x_2$', size=15)
plt.xlim([-8, 10])
plt.ylim([-8, 10])
plt.title("mu=-1.0", size = 15)
plt.show()

print("-------------")
print("Example 4")
print("-------------")

# Example 4
def f4(X, t):
    x1, x2 = X
    return [x2, -x1]

# scipy for solving IVP
t0=0
x0 = [1, 0.0]

t1=20.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f4, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-3, 3)
plt.plot(t, sol[:,0], 'k-', label='displacement', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='velocity', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# phase potrait
plt.figure(figsize=(4,4))
tspan = np.linspace(0, t1, 200)
xs = odeint(f4, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end
plt.xlabel('$x1 (displacement)$', size=15)
plt.ylabel('$x2 (velocity)$', size=15)
plt.xlim([-2, 2])
plt.ylim([-2, 2])

plt.show()


# Example 4-positive or negative
def f41(X, t):
    x1, x2 = X
    return [x2, -x1+0.1*np.sin(t)]

# scipy for solving IVP
t0=0
x0 = [1, 0.0]

t1=20.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f41, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-3, 3)
plt.plot(t, sol[:,0], 'k-', label='displacement', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='velocity', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# phase potrait
plt.figure(figsize=(4,4))
tspan = np.linspace(0, t1, 200)
xs = odeint(f41, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end
plt.xlabel('$x1 (displacement)$', size=15)
plt.ylabel('$x2 (velocity)$', size=15)
plt.xlim([-2, 2])
plt.ylim([-2, 2])

plt.show()

print("-------------")
print("Example 5")
print("-------------")

A = np.array([[0, 1], [-1, 0.2]])
le, x= spale.eig(A)   # solution from Scipy
print('the eigenvalues of matrix A are ', le)
print('the eigenvectors (columns) are \n', x)

# Example 5 linearization
def f502L(X, t):
    x1, x2 = X
    return [x2, -x1+0.2*x2]

# scipy for solving IVP
t0=0
x0 = [0.5, 0.0]

t1=50.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f502L, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-4, 4)
plt.plot(t, sol[:,0], 'k-', label='x1 (epsilon = 0.2)', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='x2', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Example 5
def f502(X, t):
    x1, x2 = X
    return [x2, -x1+0.2*(1-x1*x1)*x2]

def f510(X, t):
    x1, x2 = X
    return [x2, -x1+1.0*(1-x1*x1)*x2]


# scipy for solving IVP
t0=0
x0 = [0.5, 0.0]

t1=50.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f502, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-4, 4)
plt.plot(t, sol[:,0], 'k-', label='x1 (epsilon = 0.2)', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='x2', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# phase potrait
plt.figure(figsize=(4,4))
tspan = np.linspace(0, t1, 200)
xs = odeint(f502, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end
plt.xlabel('$x1$', size=15)
plt.ylabel('$x2$', size=15)
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.show()


# scipy for solving IVP
t0=0
x0 = [1.0, 0.0]

t1=50.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f510, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-4, 6)
plt.plot(t, sol[:,0], 'k-', label='x1 (epsilon = 1.0)', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='x2', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# phase potrait
plt.figure(figsize=(4,4))
tspan = np.linspace(0, t1, 200)
xs = odeint(f510, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end
plt.xlabel('$x1$', size=15)
plt.ylabel('$x2$', size=15)
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.show()


# scipy for solving IVP
t0=0
x0 = [3.0, 0.0]

t1=50.0
n= 1001 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f510, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-4, 6)
plt.plot(t, sol[:,0], 'k-', label='x1 (epsilon = 1.0)', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='x2', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# phase potrait
plt.figure(figsize=(4,4))
tspan = np.linspace(0, t1, 200)
xs = odeint(f510, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end
plt.xlabel('$x1$', size=15)
plt.ylabel('$x2$', size=15)
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.show()

print("-------------")
print("Example 6")
print("-------------")


# Example 6
def f60(X, t):
    x1, x2 = X
    return [x2, -x1+0.2*(1-x1*x1)*x2 - 0.1* np.sin(t)]

def f61(X, t):
    x1, x2 = X
    return [x2, -x1+0.2*(1-x1*x1)*x2 + 0.1* np.sin(t)]

# scipy for solving IVP
t0=0
x0 = [8, 0.0]

t1=50.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f61, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-5, 10)
plt.plot(t, sol[:,0], 'k-', label='x1', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='x2', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# phase potrait
plt.figure(figsize=(4,4))
tspan = np.linspace(0, t1, 200)
xs = odeint(f61, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end
plt.xlabel('$x1$', size=15)
plt.ylabel('$x2$', size=15)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.show()


print("-------------")
print("Example 7")
print("-------------")

# Example 7
def f7(X, t):
    x1, x2 = X
    return [-x2, x1-0.2*(1-x1*x1)*x2]

A = np.array([[0, -1], [1, 0.2]])
le, x= spale.eig(A)   # solution from Scipy
print('the eigenvalues of matrix A are ', le)
print('the eigenvectors (columns) are \n', x)

# scipy for solving IVP
t0=0
x0 = [2.02, 0.0]

t1=50.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f7, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-4, 4)
plt.plot(t, sol[:,0], 'k-', label='x1 (epsilon = 0.2)', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='x2', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.xlabel('Time (s)', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# phase potrait
plt.figure(figsize=(4,4))
tspan = np.linspace(0, t1, 200)
xs = odeint(f7, x0, tspan)
plt.plot(xs[:,0], xs[:,1], 'b-') # path
plt.plot([xs[0,0]], [xs[0,1]], 'o') # start
plt.plot([xs[-1,0]], [xs[-1,1]], 's') # end
plt.xlabel('$x1$', size=15)
plt.ylabel('$x2$', size=15)
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.show()


print("-------------")
print("Example 8")
print("-------------")


# Example 8
def f11(X, t):
    x1, x2 = X
    return [1-x1*x1, -x2]

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
        yprime = f11([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]
     

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x_1$', size=18)
plt.ylabel('$x_2$', size=18)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()

def f11(X, t):
    x1, x2 = X
    return [-x1*x1, -x2]

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
        yprime = f11([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]
     

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x_1$', size=18)
plt.ylabel('$x_2$', size=18)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()


def f11(X, t):
    x1, x2 = X
    return [-1-x1*x1, -x2]

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
        yprime = f11([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]
     

plt.quiver(X1, X2, u, v, color='r')
plt.xlabel('$x_1$', size=18)
plt.ylabel('$x_2$', size=18)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()


