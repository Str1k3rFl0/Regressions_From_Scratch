import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Variables
house_size = np.array([50, 70, 90, 120, 150])
price = np.array([75, 100, 130, 170, 200])

# Dependent variable y and independent variable x
x = pd.Series(house_size)
y = pd.Series(price)

# General formula: y = B0 + B1 * x
x_mean = x.mean()
y_mean = y.mean()
B1_first_sum = ((x - x_mean) * (y - y_mean)).sum()
B1_second_sum = ((x - x_mean) ** 2).sum()
B1 = B1_first_sum / B1_second_sum

# For B0
B0 = y_mean - B1 * x_mean

# Final prediction
y_pred = B0 + B1 * x

# Mean Squared Error
MSE = ((y - y_mean) ** 2).mean()
print(f"Mean Squared Error (MSE): {MSE}")

# Coefficient of determination R^2
SS_total = ((y - y_mean) ** 2).sum()
SS_residual = ((y - y_pred) ** 2).sum()
R2 = 1 - (SS_residual / SS_total)
print(f'Coefficient of Determination (R^2): {R2}')

# Request new prediction
new_size = float(input('Enter the house size for which you want to predict the price: '))
predicted_price = B0 + B1 * new_size
print(f'For a house of {new_size:.2f} m², the predicted price is {predicted_price:.2f} thousand Euros.')

# Plot
plt.scatter(x, y, color = 'blue', label = 'Original Data')
plt.scatter(x, y_pred, color = 'green', label = 'Predicted Values')
plt.scatter([new_size], [predicted_price], color = 'purple', s = 100, label = f'({new_size:.2f}, {predicted_price:.2f})')

for i in range(len(x)):
    plt.text(x[i] + 2, y[i], f'({x[i]}, {y[i]})', fontsize=9, color='blue')
    plt.text(x[i] + 2, y_pred[i], f'({x[i]}, {y_pred[i]:.2f})', fontsize=9, color='green')

plt.plot(x, y_pred, color = 'red', label = f'Regression Line\ny = {B0:.2f} + {B1:.2f}x')

# Add MSE and R^2 values to the plot
plt.text(min(x) + 10, max(y) - 10, f'MSE: {MSE:.2f}\nR²: {R2:.2f}', 
         fontsize=10, color='green', bbox=dict(facecolor='white', alpha=0.5))


plt.title('Simple Linear Regression')
plt.xlabel('House Size (m²)')
plt.ylabel('Price (thousand Euros)')
plt.legend()
plt.grid(True)
plt.show()