import matplotlib.pyplot as plt
import numpy as np

# Redefining the function for clarity
def f(x, y):
    return x**2 + y**2 + 10 * np.sin(x) + 10 * np.sin(y)

# Simulated Annealing Algorithm to find the best point again
def simulated_annealing_for_function():
    current_temp = 100.0  # Initial temperature
    min_temp = 1.0  # Minimum temperature
    alpha = 0.99  # Cooling rate
    
    current_point = np.random.rand(2) * 20 - 10  # Random initial point in [-10, 10] range
    current_value = f(*current_point)
    
    while current_temp > min_temp:
        step = (np.random.rand(2) - 0.5) * 10
        new_point = current_point + step * current_temp / 100
        new_value = f(*new_point)
        
        if new_value < current_value or np.random.rand() < np.exp((current_value - new_value) / current_temp):
            current_point = new_point
            current_value = new_value
        
        current_temp *= alpha
    
    return current_point, current_value

# Execute the algorithm
best_point, best_value = simulated_annealing_for_function()

# Generate grid for visualization
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)
z = f(x, y)

# Plotting
plt.figure(figsize=(10, 7))
contours = plt.contour(x, y, z, 50, cmap='RdGy')
plt.clabel(contours, inline=True, fontsize=8)

plt.plot(best_point[0], best_point[1], 'bo')  # Mark the minimum
plt.text(best_point[0], best_point[1], ' Minimum', verticalalignment='bottom')

plt.title('Contour plot of $f(x, y) = x^2 + y^2 + 10 \sin(x) + 10 \sin(y)$')
plt.xlabel('x')
plt.ylabel('y')
# plt.show()
plt.savefig('optim.png')
