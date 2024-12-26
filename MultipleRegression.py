import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = {
    'House_Size': [50, 70, 90, 120, 150],
    'Number_of_Rooms': [2, 3, 3, 4, 5],
    'Price': [75, 100, 130, 170, 200]
}

df = pd.DataFrame(data)

# Independent variables (X)
# Dependent variable (y)
X = df[['House_Size', 'Number_of_Rooms']]
y = df['Price']

array_x = np.c_[np.ones(X.shape[0]), X]
array_y = np.array(y)
print('Matrix X:\n', array_x)
print('Matrix Y:\n', array_y)

# Calculate coefficients
trans_array_x = np.transpose(array_x)
produs_xtrans_x = np.dot(trans_array_x, array_x)
inversa_prod_arrayX = np.linalg.inv(produs_xtrans_x)
produs_xtrans_y = np.dot(trans_array_x, array_y)
B = np.dot(inversa_prod_arrayX, produs_xtrans_y)

# Prediction
y_pred = X.dot(B[1:]) + B[0]

# Calculate MSE
MSE = ((y - y_pred) ** 2).mean()
print(f'Mean Squared Error (MSE): {MSE:.2f}')

# Calculate R^2
SS_total = ((y - y.mean()) ** 2).sum()
SS_residual = ((y - y_pred) ** 2).sum()
R2 = 1 - (SS_residual / SS_total)
print(f'Coefficient of Determination (R^2): {R2}')

# User enters data for prediction
print('\nEnter the data to predict the house price:')
house_size_new = float(input('House Size (m²): '))
number_of_rooms_new = int(input('Number of Rooms: '))
new_prediction = B[0] + B[1] * house_size_new + B[2] * number_of_rooms_new
print(f'The prediction for the house price is: {new_prediction:.2f} thousand Euros')

# Plot
B0 = B[0]
B1 = B[1]
B2 = B[2]
equation = f'Y = {B0:.2f} + {B1:.2f}*x1 + {B2:.2f}*x2'

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(df['House_Size'], df['Number_of_Rooms'], df['Price'], color = 'blue', label = 'Real Data')
ax.scatter(df['House_Size'], df['Number_of_Rooms'], y_pred, color = 'green', label = f'Predictions\n{equation}')
ax.scatter(house_size_new, number_of_rooms_new, new_prediction, color = 'black', s = 20, label = f'New Prediction\n{new_prediction:.2f} thousand Euros')

for i in range(len(df)):
    ax.text(df['House_Size'][i], df['Number_of_Rooms'][i], df['Price'][i], 
            f'({df["House_Size"][i]}, {df["Number_of_Rooms"][i]}, {df["Price"][i]})', color='blue')

for i in range(len(df)):
    ax.text(df['House_Size'][i], df['Number_of_Rooms'][i], y_pred[i], 
            f'({df["House_Size"][i]}, {df["Number_of_Rooms"][i]}, {y_pred[i]:.2f})', color='red')

ax.set_xlabel('House Size (m²)')
ax.set_ylabel('Number of Rooms')
ax.set_zlabel('Price (thousand Euros)')
ax.set_title('Multiple Linear Regression')
ax.legend()

plt.show()
