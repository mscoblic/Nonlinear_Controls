# ME:5114 Nonlinear Control
# Lecture 4: stability of linear systems
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
    return -x1**3.0

# initial displacement
t0 = 0.0
x0 = 1

# scipy for solving IVP
t1=100.0
n= 1001 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol1 = odeint(f1, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-0.2, 1.2)
plt.plot(t, sol1, 'k-', label='x', linewidth =2)
plt.plot(t, t*0, 'k--', label='x=0', linewidth =2)
plt.legend( loc='best', scatterpoints=1,  fontsize=12, frameon=False, labelspacing=0.5)
plt.ylabel("x(t)", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

print("-------------")
print("Example 2")
print("-------------")


# Example 2
def f2(x, t):
    x1 = x
    return -0.5*x1

# initial displacement
t0 = 0.0
x0 = 1

# scipy for solving IVP
t1=100.0
n= 1001 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol2 = odeint(f2, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-0.2, 1.2)
plt.plot(t, sol1, 'k-', label='Asymptotically stable (Example 1)', linewidth =2)
plt.plot(t, sol2, 'k-.', label='exponentially stable (Example 2)', linewidth =2)
plt.plot(t, t*0, 'k--', label='x=0', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=12, frameon=False, labelspacing=0.5)
plt.ylabel("x(t)", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()



print("-------------")
print("Example 3")
print("-------------")


# Example 3

import numpy as np
from scipy.integrate import odeint

def f(x, t):
    x1, x2 = x
    return [x2, -10*x1]

# initial condition
t0 = 0.0
x0 = [1.0, 2.0]

# scipy for solving IVP
t1=10.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f, x0, t)


plt.figure(figsize=(6,3))
plt.ylim(-4, 8)
plt.plot(t, sol[:,0], 'k-', label='displacement', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='velocity', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1 or x2", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("initial condition (x1=1, x2=2)", size=15)
plt.show()



plt.figure(figsize=(4,4))
plt.ylim(-5, 5)
plt.xlim(-5, 5)
plt.plot(sol[:,0], sol[:,1], 'k-', label='phase portrait', linewidth =2)
plt.legend( loc='upper left', scatterpoints=1,  fontsize=12, frameon=False, labelspacing=0.5)
plt.ylabel("x2 (m/s)", size=18)
plt.xlabel('x1 (m)', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# eigen values

A = np.array([[0, 1], [-10, 0]])
le, x= spale.eig(A)   # solution from Scipy
print('the eigenvalues of matrix A are ', le)
print('the eigenvectors (columns) are \n', x)

print("-------------")
print("Example 4")
print("-------------")



# Example 4
def f4(x, t):
    x1, x2 = x
    return [x2, -10*x1-x2]

# initial condition
t0 = 0.0
x0 = [1.0, 2.0]

# scipy for solving IVP
t1=10.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f4, x0, t)


plt.figure(figsize=(6,3))
plt.ylim(-4, 4)
plt.plot(t, sol[:,0], 'k-', label='displacement', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='velocity', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1 or x2", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("initial condition (x1=1, x2=2)", size=15)
plt.show()


plt.figure(figsize=(4,4))
plt.ylim(-3, 3)
plt.xlim(-3, 3)
plt.plot(sol[:,0], sol[:,1], 'k-', label='phase portrait', linewidth =2)
plt.legend( loc='upper left', scatterpoints=1,  fontsize=12, frameon=False, labelspacing=0.5)
plt.ylabel("x2 (m/s)", size=18)
plt.xlabel('x1 (m)', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# eigen values

A = np.array([[0, 1], [-10, -1]])
le, x= spale.eig(A)   # solution from Scipy
print('the eigenvalues of matrix A are ', le)
print('the eigenvectors (columns) are \n', x)

print("-------------")
print("Example 5")
print("-------------")

# Example 5
def f5(x, t):
    x1, x2 = x
    return [x2, -10*x1+x2]

# initial condition
t0 = 0.0
x0 = [1.0, 2.0]

# scipy for solving IVP
t1=10.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f5, x0, t)


plt.figure(figsize=(6,3))
plt.ylim(-200, 200)
plt.plot(t, sol[:,0], 'k-', label='displacement', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='velocity', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1 or x2", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("initial condition (x1=1, x2=2)", size=15)
plt.show()

plt.figure(figsize=(4,4))
plt.ylim(-500, 500)
plt.xlim(-500, 500)
plt.plot(sol[:,0], sol[:,1], 'k-', label='phase portrait', linewidth =2)
plt.legend( loc='lower right', scatterpoints=1,  fontsize=12, frameon=False, labelspacing=0.5)
plt.ylabel("x2 (m/s)", size=18)
plt.xlabel('x1 (m)', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# eigen values

A = np.array([[0, 1], [-10, 1]])
le, x= spale.eig(A)   # solution from Scipy
print('the eigenvalues of matrix A are ', le)
print('the eigenvectors (columns) are \n', x)

print("-------------")
print("Example 6")
print("-------------")

# Example 6

def f6(x, t):
    x1, x2 = x
    u=x2
    return [x2, -10*x1+x2-u]

def f6i(x, t):
    x1, x2 = x
    return [x2, -10*x1+x2]

A = np.array([[0, 1], [-10, 1]])
le, x= spale.eig(A)   # solution from Scipy
print('the eigenvalues of matrix A are ', le)
print('the eigenvectors (columns) are \n', x)


# initial condition
t0 = 0.0
x0 = [1.0, 0.0]

# scipy for solving IVP
t1=10.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at

soli = odeint(f6i, x0, t)

plt.figure(figsize=(6,3))
plt.ylim(-100, 100)
plt.plot(t, t*0, 'k-.', linewidth =2)
plt.plot(t, soli[:,0], 'k--', label='zero-input response', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

x0 = [1.0, 0.0]
sol = odeint(f6, x0, t)
plt.figure(figsize=(6,3))
plt.ylim(-5, 10)
plt.plot(t, sol[:,1], 'k-.', label='x2 (feed back as input signal)', linewidth =2)
plt.plot(t, sol[:,0], 'k-', label='x1', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1,x2", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

print("-------------")
print("Example 7")
print("-------------")

# Example 7

a = 10.0
def f7(x, t):
    x1, x2 = x
    return [x2, -a*a*x1-2*a*x2]

# initial condition
t0 = 0.0
x0 = [1.0, 0]

# scipy for solving IVP
t1=1.0
n= 101 # number of time points
dt = (t1-t0)/(n-1)
t = np.linspace(t0, t1, n) # points where solutions are solved at
sol = odeint(f7, x0, t)
  
plt.figure(figsize=(6,3))
plt.ylim(-4, 2)
plt.plot(t, sol[:,0], 'k-', label='x1', linewidth =2)
plt.plot(t, sol[:,1], 'k-.', label='x2', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.ylabel("x1 or x2", size=18)
plt.xlabel('Time', size=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("initial condition (x1=1, x2=0)", size=15)
plt.show()
    

y=np.zeros(len(t))
y[0]=10
y0=y[0]
for i in range (1, n):
    y[i]=y0+dt*(-(1+sol[i,1])/2*y0*y0*y0)
    y0 = y[i]
     
plt.figure(figsize=(6,3))
plt.xlim(0, 1.0)
plt.plot(t, y, 'k-', label='y (y0=10)', linewidth =2)
plt.plot(t, t*0, 'k--', linewidth =1)
plt.plot(0.5*np.ones(len(t)), t*0, 'k--', linewidth =1)
plt.legend( loc='best', scatterpoints=1,  fontsize=12, frameon=False, labelspacing=0.5)
plt.ylabel("y", size=15)
plt.xlabel('Time', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()








