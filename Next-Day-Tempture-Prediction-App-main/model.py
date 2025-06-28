# Prepare a time series model for Prediction of next Day Tremerature
# Import the libaries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the dataset into the DataFrame
#parse_date : is used to convert the date column into datetime objects
df = pd.read_csv("daily_minimum_temps.csv", parse_dates = ["Date"], index_col = "Date")

df.head()
df.isna().sum(axis=0)

# Converting the string values to Numeric value by removing the double quotes and if string value does not make any sense or it
# We convert that value to null.
df["Temp"] = pd.to_numeric(df["Temp"], errors = "coerce")

#df["Temp"]=df["Temp"].dropna()
df = df.dropna()

# Normalize the features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df["Temp"].values.reshape(-1,1))

# Sequence Length for temperature
seq_length = 30

# Function for creating sequences
def create_sequences(data_scaled,seq_length):
  x,y=[],[]
  for i in range (len(data_scaled)-seq_length):
    x.append(data_scaled[i:i+seq_length])
    y.append(data_scaled[i+seq_length])
  return np.array(x),np.array(y)

# Calling the function and storing the values in list x and y
x,y = create_sequences(data_scaled,seq_length)

# Divide the data set into train and test and split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 1, shuffle = False)

# Building the RNN Model
model = Sequential([
    LSTM(64,activation = "relu",input_shape = (seq_length,1)),
    Dense(1) # Since Output will be a single value
])

#Compile the Model
model.compile(optimizer = "adam", loss = "mse")

#Train the model
model.fit(x_train,y_train, epochs = 20, batch_size = 32)

#Make Prediction
y_pred_scaled = model.predict(x_test)

#Inverse transform the scaled data
y_pred_scaled = np.clip(y_pred_scaled,0,1)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)

# predict the next day temperature
last_sequence = data_scaled[-seq_length:].reshape(1,seq_length,1)
next_temp_scaled= model.predict(last_sequence)
next_temp_scaled= np.clip(next_temp_scaled, 0, 1)
next_temp = scaler.inverse_transform(next_temp_scaled)
print("The next day's temperature is: ", next_temp)


joblib.dump(model, "next_day_temp.pkl")
joblib.dump(scaler, "scaler.pkl")
