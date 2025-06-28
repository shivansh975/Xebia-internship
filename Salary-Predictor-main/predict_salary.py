# A simple app to predict the salary based on the number of years of experience
# Importing the required libraries
# joblib : Library to save and load the create the model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# Loding the dataset into the dataframe
df = pd.read_csv("salary_data.csv")
print(df.info())

# Split the data into target variable and Independent Variables
x = df[["YearsExperience"]]
y = df[["Salary"]]

# Train -test-split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)

# Scale down the data
# Creating an object of StandardScaler Module present in sklearn libaries
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

#Train the Model
model = LinearRegression()
model.fit(x_train_scaled,y_train)

# Save the Model
joblib.dump(model,"predict_salary.pk1")
joblib.dump(scaler,"scaler.pk1")
print("Model and Scaler are saved") 