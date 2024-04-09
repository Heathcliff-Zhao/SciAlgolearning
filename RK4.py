import matplotlib.pyplot as plt
import numpy as np

# Given t_values and y_values from the previous example
t_values = [0]
y_values = [1]
y0 = 1
t0 = 0
tf = 5
h = 0.01  # Step size
a = -1  # Constant in the differential equation

# Defining the analytical solution
def af(t):
    return y0 * np.e**(-t)

# Redefining f for local context
def f(t, y, a=-1):
    return a * y

# Redefining rk4_step for local context
def rk4_step(t, y, h, a=-1):
    k1 = f(t, y, a)
    k2 = f(t + h/2, y + h/2 * k1, a)
    k3 = f(t + h/2, y + h/2 * k2, a)
    k4 = f(t + h, y + h * k3, a)
    y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_next

# Solving the ODE
t = t0
y = y0
while t < tf:
    y = rk4_step(t, y, h, a)
    t += h
    t_values.append(t)
    y_values.append(y)

# Calculating the analytical solution
y_analytical = [af(t) for t in t_values]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_values, label='RK4 Solution', color='blue')
plt.plot(t_values, y_analytical, label='Analytical Solution', color='red', linestyle='dashed')
plt.title('Solution of dy/dt = -y using RK4 Method')
plt.xlabel('Time (t)')
plt.ylabel('y(t)')
plt.grid(True)
plt.legend()
# plt.show()
plt.savefig('RK4.png')
