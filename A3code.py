#!/usr/bin/env python
# coding: utf-8

# # Import

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import SimpleRNN, Bidirectional, LSTM, Dense


# # Data Preprocessing

# In[43]:


# Load data
train_data = pd.read_csv('Google_Stock_Price_Train.csv')
test_data = pd.read_csv('Google_Stock_Price_Test.csv')

# Convert dates to datetime format
train_data['Date'] = pd.to_datetime(train_data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])

# Process price columns
def preprocess_price_columns(df, columns):
    for column in columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.replace(',', '').astype(float)
    return df

price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
train_data = preprocess_price_columns(train_data, price_columns)
test_data = preprocess_price_columns(test_data, price_columns)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_data[price_columns].values)
scaled_test = scaler.transform(test_data[price_columns].values)

# Create sliding window dataset
def create_multivariate_dataset(data, time_steps=30):
    if len(data) <= time_steps:
        raise ValueError(f"Data length ({len(data)}) is too short for the specified time_steps ({time_steps}).")
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, :])
        y.append(data[i + time_steps, :])  # Predict the next day
    return np.array(X), np.array(y)

time_steps = 30  # Use the past 30 days
X_train, y_train = create_multivariate_dataset(scaled_train, time_steps)

# Fix insufficient test data
total_data = np.concatenate((scaled_train, scaled_test), axis=0)
inputs = total_data[len(total_data) - len(test_data) - time_steps:]
X_test, y_test = create_multivariate_dataset(inputs, time_steps)

# Debug print shapes
print(f"X_train.shape: {X_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"y_test.shape: {y_test.shape}")

# Reshape data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))


# # Data visualizations

# In[44]:


# Plot histogram of closing prices in the training set
plt.figure(figsize=(10, 6))
plt.hist(train_data['Close'], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title('Frequency Distribution of Training Set Closing Prices')
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plot histogram of closing prices in the test set
plt.figure(figsize=(10, 6))
plt.hist(test_data['Close'], bins=30, color='orange', edgecolor='black', alpha=0.7)
plt.title('Frequency Distribution of Test Set Closing Prices')
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plot the trend of closing prices in the training set
plt.figure(figsize=(14, 8))
plt.plot(train_data['Date'], train_data['Close'], color='blue', label='Training Set Close Price')
plt.title('Training Set Closing Price Trend')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid()
plt.show()

# Plot the trend of closing prices in the test set
plt.figure(figsize=(14, 8))
plt.plot(test_data['Date'], test_data['Close'], color='orange', label='Test Set Close Price')
plt.title('Test Set Closing Price Trend')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid()
plt.show()


# # RNN1

# In[8]:


# Build the RNN model
rnn_model = Sequential([
    SimpleRNN(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=X_train.shape[2])  # Output dimension equal to the number of features
])
rnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Train the model
history = rnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Predict using the model
y_pred = rnn_model.predict(X_test)

# Inverse scaling
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Combine the dates of training and testing datasets
total_dates = pd.concat([train_data['Date'], test_data['Date']], ignore_index=True)

# Retrieve corresponding dates
date_inputs = total_dates.iloc[len(total_dates) - len(test_data) - time_steps:]
plot_dates = date_inputs.iloc[time_steps:].reset_index(drop=True)

# Debug print statements
print(f"plot_dates length: {len(plot_dates)}")
print(f"y_test_actual length: {len(y_test_actual)}")

# Plot the actual vs predicted closing prices (with dates)
plt.figure(figsize=(14, 7))
plt.plot(plot_dates, y_test_actual[:, 3], label='Actual Close Price', color='blue')
plt.plot(plot_dates, y_pred[:, 3], label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price (Next Day)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Set date format and interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show date every 2 days

# Automatically adjust date labels to avoid overlapping
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

# Plot the training loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Calculate MAE
mae = mean_absolute_error(y_test_actual[:, 3], y_pred[:, 3])
print(f"MAE: {mae}")


# # RNN2

# In[11]:


# Build the RNN model
rnn_model = Sequential([
    SimpleRNN(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=X_train.shape[2])  # Output dimensions equal to the number of features
])
rnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Callback for dynamic learning rate adjustment
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Increase the number of epochs
epochs = 50  # Increase training epochs to 50

# Train the model
history = rnn_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=32,
    validation_split=0.2,
    callbacks=[reduce_lr],  # Use learning rate scheduler
    verbose=1
)

# Model prediction
y_pred = rnn_model.predict(X_test)

# Inverse scaling
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Retrieve corresponding dates
total_dates = pd.concat([train_data['Date'], test_data['Date']], ignore_index=True)
date_inputs = total_dates.iloc[len(total_dates) - len(test_data) - time_steps:]
plot_dates = date_inputs.iloc[time_steps:].reset_index(drop=True)

# Plot actual vs predicted closing prices (with dates)
plt.figure(figsize=(14, 7))
plt.plot(plot_dates, y_test_actual[:, 3], label='Actual Close Price', color='blue')
plt.plot(plot_dates, y_pred[:, 3], label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price (Next Day)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Set date format and interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show date every 2 days

# Automatically adjust date labels to avoid overlapping
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

# Plot loss curve during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Compute MAE
mae = mean_absolute_error(y_test_actual[:, 3], y_pred[:, 3])
print(f"MAE: {mae}")

# Residual analysis
residuals = y_test_actual[:, 3] - y_pred[:, 3]
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# # RNN3

# In[12]:


# Build the RNN model
rnn_model = Sequential([
    SimpleRNN(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=X_train.shape[2])  # Output dimensions equal to the number of features
])
rnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Callback for dynamic learning rate adjustment
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Increase the number of epochs
epochs = 100  # Increase training epochs to 100

# Train the model
history = rnn_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=32,
    validation_split=0.2,
    callbacks=[reduce_lr],  # Use learning rate scheduler
    verbose=1
)

# Model prediction
y_pred = rnn_model.predict(X_test)

# Inverse scaling
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Retrieve corresponding dates
total_dates = pd.concat([train_data['Date'], test_data['Date']], ignore_index=True)
date_inputs = total_dates.iloc[len(total_dates) - len(test_data) - time_steps:]
plot_dates = date_inputs.iloc[time_steps:].reset_index(drop=True)

# Plot actual vs predicted closing prices (with dates)
plt.figure(figsize=(14, 7))
plt.plot(plot_dates, y_test_actual[:, 3], label='Actual Close Price', color='blue')
plt.plot(plot_dates, y_pred[:, 3], label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price (Next Day)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Set date format and interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show date every 2 days

# Automatically adjust date labels to avoid overlapping
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

# Plot loss curve during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Compute MAE
mae = mean_absolute_error(y_test_actual[:, 3], y_pred[:, 3])
print(f"MAE: {mae}")

# Residual analysis
residuals = y_test_actual[:, 3] - y_pred[:, 3]
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# # RNN4

# In[16]:


# Build the RNN model
rnn_model = Sequential([
    SimpleRNN(units=128, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),  # Change the number of neurons to 128
    Dense(units=X_train.shape[2])  # Output dimensions equal to the number of features
])
rnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Callback for dynamic learning rate adjustment
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Number of training epochs
epochs = 100  # Set the number of training epochs to 100

# Train the model
history = rnn_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=32,
    validation_split=0.2,
    callbacks=[reduce_lr],  # Use learning rate scheduler
    verbose=1
)

# Model prediction
y_pred = rnn_model.predict(X_test)

# Inverse scaling
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Retrieve corresponding dates
total_dates = pd.concat([train_data['Date'], test_data['Date']], ignore_index=True)
date_inputs = total_dates.iloc[len(total_dates) - len(test_data) - time_steps:]
plot_dates = date_inputs.iloc[time_steps:].reset_index(drop=True)

# Plot actual vs predicted closing prices (with dates)
plt.figure(figsize=(14, 7))
plt.plot(plot_dates, y_test_actual[:, 3], label='Actual Close Price', color='blue')
plt.plot(plot_dates, y_pred[:, 3], label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price (Next Day)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Set date format and interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show date every 2 days

# Automatically adjust date labels to avoid overlapping
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

# Plot the loss curve during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Calculate MAE
mae = mean_absolute_error(y_test_actual[:, 3], y_pred[:, 3])
print(f"MAE: {mae}")

# Residual analysis
residuals = y_test_actual[:, 3] - y_pred[:, 3]
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# # GRU1

# In[17]:


from tensorflow.keras.layers import GRU

# Build the GRU model
gru_model = Sequential([
    GRU(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),  # Use GRU and set 50 neurons
    Dense(units=X_train.shape[2])  # Output dimensions equal to the number of features
])
gru_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Train the model
history = gru_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Model predictions
y_pred = gru_model.predict(X_test)

# Inverse scaling
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Combine training and testing dates
total_dates = pd.concat([train_data['Date'], test_data['Date']], ignore_index=True)

# Get corresponding dates
date_inputs = total_dates.iloc[len(total_dates) - len(test_data) - time_steps:]
plot_dates = date_inputs.iloc[time_steps:].reset_index(drop=True)

# Debug print
print(f"plot_dates length: {len(plot_dates)}")
print(f"y_test_actual length: {len(y_test_actual)}")

# Plot actual vs predicted closing prices (with dates)
plt.figure(figsize=(14, 7))
plt.plot(plot_dates, y_test_actual[:, 3], label='Actual Close Price', color='blue')
plt.plot(plot_dates, y_pred[:, 3], label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price (Next Day)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Set date format and interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show a date every 2 days

# Automatically adjust date labels to avoid overlap
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

# Plot the loss curve during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Calculate MAE
mae = mean_absolute_error(y_test_actual[:, 3], y_pred[:, 3])
print(f"MAE: {mae}")


# # GRU2

# In[19]:


from tensorflow.keras.callbacks import ReduceLROnPlateau

# Build the GRU model
gru_model = Sequential([
    GRU(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),  # Use GRU and set 50 neurons
    Dense(units=X_train.shape[2])  # Output dimensions equal to the number of features
])
gru_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Callback for dynamic learning rate adjustment
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Adjust the number of training epochs
epochs = 50  # Set the number of training epochs to 50

# Train the model
history = gru_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=32,
    validation_split=0.2,
    callbacks=[reduce_lr],  # Use learning rate scheduler
    verbose=1
)

# Model predictions
y_pred = gru_model.predict(X_test)

# Inverse scaling
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Combine training and testing dates
total_dates = pd.concat([train_data['Date'], test_data['Date']], ignore_index=True)

# Retrieve corresponding dates
date_inputs = total_dates.iloc[len(total_dates) - len(test_data) - time_steps:]
plot_dates = date_inputs.iloc[time_steps:].reset_index(drop=True)

# Debug print
print(f"plot_dates length: {len(plot_dates)}")
print(f"y_test_actual length: {len(y_test_actual)}")

# Plot actual vs predicted closing prices (with dates)
plt.figure(figsize=(14, 7))
plt.plot(plot_dates, y_test_actual[:, 3], label='Actual Close Price', color='blue')
plt.plot(plot_dates, y_pred[:, 3], label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price (Next Day)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Set date format and interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show a date every 2 days

# Automatically adjust date labels to avoid overlap
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

# Plot the loss curve during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training (GRU)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Calculate MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_actual[:, 3], y_pred[:, 3])
print(f"MAE: {mae}")

# Residual analysis
residuals = y_test_actual[:, 3] - y_pred[:, 3]
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# # GRU3

# In[22]:


from tensorflow.keras.callbacks import ReduceLROnPlateau

# Build the GRU model
gru_model = Sequential([
    GRU(units=128, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),  # GRU layer with 128 units
    Dense(units=X_train.shape[2])  # Output dimensions equal to the number of features
])
gru_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Callback for dynamic learning rate adjustment
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Set the number of training epochs to 50
epochs = 50

# Train the model
history = gru_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=32,
    validation_split=0.2,
    callbacks=[reduce_lr],  # Use learning rate scheduler
    verbose=1
)

# Model prediction
y_pred = gru_model.predict(X_test)

# Inverse scaling
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Combine training and testing dates
total_dates = pd.concat([train_data['Date'], test_data['Date']], ignore_index=True)

# Retrieve corresponding dates
date_inputs = total_dates.iloc[len(total_dates) - len(test_data) - time_steps:]
plot_dates = date_inputs.iloc[time_steps:].reset_index(drop=True)

# Plot actual vs predicted closing prices (with dates)
plt.figure(figsize=(14, 7))
plt.plot(plot_dates, y_test_actual[:, 3], label='Actual Close Price', color='blue')
plt.plot(plot_dates, y_pred[:, 3], label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price (Next Day)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Set date format and interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show a date every 2 days

# Automatically adjust date labels to avoid overlap
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

# Plot the loss curve during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training (GRU)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Calculate MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_actual[:, 3], y_pred[:, 3])
print(f"MAE: {mae}")

# Residual analysis
residuals = y_test_actual[:, 3] - y_pred[:, 3]
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# # GRU4

# In[24]:


from tensorflow.keras.callbacks import ReduceLROnPlateau

# Build the GRU model
gru_model = Sequential([
    GRU(units=128, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),  # GRU layer
    Dense(units=X_train.shape[2])  # Output dimensions equal to the number of features
])
gru_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Callback for dynamic learning rate adjustment
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Set the number of training epochs to 100
epochs = 100

# Train the model
history = gru_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=64,
    validation_split=0.2,
    callbacks=[reduce_lr],  # Use learning rate scheduler
    verbose=1
)

# Model prediction
y_pred = gru_model.predict(X_test)

# Inverse scaling
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Combine training and testing dates
total_dates = pd.concat([train_data['Date'], test_data['Date']], ignore_index=True)

# Retrieve corresponding dates
date_inputs = total_dates.iloc[len(total_dates) - len(test_data) - time_steps:]
plot_dates = date_inputs.iloc[time_steps:].reset_index(drop=True)

# Plot actual vs predicted closing prices (with dates)
plt.figure(figsize=(14, 7))
plt.plot(plot_dates, y_test_actual[:, 3], label='Actual Close Price', color='blue')
plt.plot(plot_dates, y_pred[:, 3], label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price (Next Day)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Set date format and interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show a date every 2 days

# Automatically adjust date labels to avoid overlap
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

# Plot the loss curve during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training (GRU)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Calculate MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_actual[:, 3], y_pred[:, 3])
print(f"MAE: {mae}")

# Residual analysis
residuals = y_test_actual[:, 3] - y_pred[:, 3]
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# # LSTM1

# In[26]:


from tensorflow.keras.layers import LSTM

# Build the LSTM model
lstm_model = Sequential([
    LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),  # Use LSTM with 50 neurons
    Dense(units=X_train.shape[2])  # Output dimensions equal to the number of features
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Train the model
history = lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Model prediction
y_pred = lstm_model.predict(X_test)

# Inverse scaling
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Combine training and testing dates
total_dates = pd.concat([train_data['Date'], test_data['Date']], ignore_index=True)

# Retrieve corresponding dates
date_inputs = total_dates.iloc[len(total_dates) - len(test_data) - time_steps:]
plot_dates = date_inputs.iloc[time_steps:].reset_index(drop=True)

# Debugging print
print(f"plot_dates length: {len(plot_dates)}")
print(f"y_test_actual length: {len(y_test_actual)}")

# Plot actual vs predicted closing prices (with dates)
plt.figure(figsize=(14, 7))
plt.plot(plot_dates, y_test_actual[:, 3], label='Actual Close Price', color='blue')
plt.plot(plot_dates, y_pred[:, 3], label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price (Next Day)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Set date format and interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show a date every 2 days

# Automatically adjust date labels to avoid overlap
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

# Plot the loss curve during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Calculate MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_actual[:, 3], y_pred[:, 3])
print(f"MAE: {mae}")


# # LSTM2

# In[28]:


from tensorflow.keras.callbacks import ReduceLROnPlateau

# Build the LSTM model
lstm_model = Sequential([
    LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),  # Use LSTM with 50 neurons
    Dense(units=X_train.shape[2])  # Output dimensions equal to the number of features
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Callback function to dynamically adjust the learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Increase training epochs
epochs = 50  # Increase the number of training epochs to 50

# Train the model
history = lstm_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=32,
    validation_split=0.2,
    callbacks=[reduce_lr],  # Use learning rate scheduler
    verbose=1
)

# Model prediction
y_pred = lstm_model.predict(X_test)

# Inverse scaling
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Combine training and testing dates
total_dates = pd.concat([train_data['Date'], test_data['Date']], ignore_index=True)

# Retrieve corresponding dates
date_inputs = total_dates.iloc[len(total_dates) - len(test_data) - time_steps:]
plot_dates = date_inputs.iloc[time_steps:].reset_index(drop=True)

# Debugging print
print(f"plot_dates length: {len(plot_dates)}")
print(f"y_test_actual length: {len(y_test_actual)}")

# Plot actual vs predicted closing prices (with dates)
plt.figure(figsize=(14, 7))
plt.plot(plot_dates, y_test_actual[:, 3], label='Actual Close Price', color='blue')
plt.plot(plot_dates, y_pred[:, 3], label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price (Next Day)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Set date format and interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show a date every 2 days

# Automatically adjust date labels to avoid overlap
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

# Plot the loss curve during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Calculate MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_actual[:, 3], y_pred[:, 3])
print(f"MAE: {mae}")

# Residual analysis
residuals = y_test_actual[:, 3] - y_pred[:, 3]
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# # LSTM3

# In[33]:


from tensorflow.keras.callbacks import ReduceLROnPlateau

# Build LSTM model
lstm_model = Sequential([
    LSTM(units=128, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),  # LSTM with 128 neurons
    Dense(units=X_train.shape[2])  # Output dimension matches the number of features
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Callback for dynamically adjusting the learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Increase the number of training epochs
epochs = 50  # Set the number of training epochs to 50

# Train the model
history = lstm_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=64,  # Batch size set to 64
    validation_split=0.2,
    callbacks=[reduce_lr],  # Use learning rate scheduler
    verbose=1
)

# Predict with the model
y_pred = lstm_model.predict(X_test)

# Reverse scaling
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Combine training and testing dates
total_dates = pd.concat([train_data['Date'], test_data['Date']], ignore_index=True)

# Extract corresponding dates
date_inputs = total_dates.iloc[len(total_dates) - len(test_data) - time_steps:]
plot_dates = date_inputs.iloc[time_steps:].reset_index(drop=True)

# Debugging print
print(f"plot_dates length: {len(plot_dates)}")
print(f"y_test_actual length: {len(y_test_actual)}")

# Plot the predicted vs actual closing prices (with dates)
plt.figure(figsize=(14, 7))
plt.plot(plot_dates, y_test_actual[:, 3], label='Actual Close Price', color='blue')
plt.plot(plot_dates, y_pred[:, 3], label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price (Next Day)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Set date format and interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Display dates every 2 days

# Automatically adjust date labels to avoid overlap
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

# Plot the loss curve during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Compute MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_actual[:, 3], y_pred[:, 3])
print(f"MAE: {mae}")

# Residual analysis
residuals = y_test_actual[:, 3] - y_pred[:, 3]
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# # LSTM4

# In[37]:


from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional

# Build the improved Bidirectional LSTM model
lstm_model = Sequential([
    Bidirectional(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),  # Bidirectional LSTM
    LSTM(units=32, return_sequences=False),  # Second LSTM layer
    Dense(units=X_train.shape[2])  # Output layer matching the number of features
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Adjust learning rate and train for 50 epochs
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[reduce_lr], verbose=1)

# Model predictions
y_pred = lstm_model.predict(X_test)

# Reverse normalization
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Combine training and test dates
total_dates = pd.concat([train_data['Date'], test_data['Date']], ignore_index=True)

# Retrieve corresponding dates
date_inputs = total_dates.iloc[len(total_dates) - len(test_data) - time_steps:]
plot_dates = date_inputs.iloc[time_steps:].reset_index(drop=True)

# Debugging print
print(f"plot_dates length: {len(plot_dates)}")
print(f"y_test_actual length: {len(y_test_actual)}")

# Plot predicted vs actual closing prices (with dates)
plt.figure(figsize=(14, 7))
plt.plot(plot_dates, y_test_actual[:, 3], label='Actual Close Price', color='blue')
plt.plot(plot_dates, y_pred[:, 3], label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price (Next Day)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Set date format and interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Display dates every 2 days

# Automatically adjust date labels to avoid overlap
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

# Plot loss curves during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Calculate MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_actual[:, 3], y_pred[:, 3])
print(f"MAE: {mae}")

# Residual analysis
residuals = y_test_actual[:, 3] - y_pred[:, 3]
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# # LSTM5

# In[41]:


from tensorflow.keras.callbacks import ReduceLROnPlateau

# Build the three-layer stacked Bidirectional LSTM model
from tensorflow.keras.layers import Bidirectional

stacked_lstm_model = Sequential([
    Bidirectional(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),  # First Bidirectional LSTM layer
    Bidirectional(LSTM(units=64, return_sequences=True)),  # Second Bidirectional LSTM layer
    Bidirectional(LSTM(units=64, return_sequences=False)),  # Third Bidirectional LSTM layer
    Dense(units=X_train.shape[2])  # Output layer, with the number of features equal to the target dimensions
])

# Compile the model
stacked_lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Dynamic learning rate adjustment callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Increase the number of training epochs
epochs = 50  # Number of epochs
batch_size = 32  # Batch size

# Train the model
history = stacked_lstm_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[reduce_lr],  # Use the learning rate scheduler
    verbose=1
)

# Model prediction
y_pred = stacked_lstm_model.predict(X_test)

# Inverse normalization
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Combine the dates from the training and testing datasets
total_dates = pd.concat([train_data['Date'], test_data['Date']], ignore_index=True)
date_inputs = total_dates.iloc[len(total_dates) - len(test_data) - time_steps:]
plot_dates = date_inputs.iloc[time_steps:].reset_index(drop=True)

# Plot the comparison of predicted vs. actual closing prices (with dates)
plt.figure(figsize=(14, 7))
plt.plot(plot_dates, y_test_actual[:, 3], label='Actual Close Price', color='blue')
plt.plot(plot_dates, y_pred[:, 3], label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price (Next Day)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Set date format and interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Display a date every 2 days

# Automatically adjust date labels to avoid overlapping
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

# Plot the loss curves during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Calculate MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_actual[:, 3], y_pred[:, 3])
print(f"MAE: {mae}")

# Residual analysis
residuals = y_test_actual[:, 3] - y_pred[:, 3]
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()

