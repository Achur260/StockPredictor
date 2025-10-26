from pyexpat import model

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out




def get_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

# data = get_stock_data("GOOG", "2020-01-01", "2025-01-01")
# print(data.head())
#
# data.to_csv("GOOG.csv")



df = get_stock_data('AMZN', start_date='2020-1-01', end_date='2025-1-28')
print(df.head())
print(df.columns)
print(df.index)
# df.rename({"Price":"Date"}, axis=1, inplace=True)
# df["Date"] = pd.to_datetime(df["Date"])
print(df.dtypes)
print(df.columns)

df.index = pd.to_datetime(df.index)
df['Prev_Open'] = df["Open"].shift(1)
df['Prev_High'] = df["High"].shift(1)
df['Prev_Low'] = df["Low"].shift(1)
df['Prev_Close'] = df["Close"].shift(1)
df['Prev_Volume'] = df["Volume"].shift(1)
df['Month'] = df.index.month
df['Year'] = df.index.year
df['DayOfWeek'] = df.index.dayofweek
df['MA_10'] = df["Close"].shift(1).rolling(10).mean()
df['MA_50'] = df["Close"].shift(1).rolling(50).mean()
df['EMA_10'] = df['Close'].shift(1).ewm(span=10, adjust=False).mean()
df['EMA_50'] = df['Close'].shift(1).ewm(span=50, adjust=False).mean()
df['EMA_5'] = df['Close'].shift(1).ewm(span=5, adjust=False).mean()
df["MA_5"] = df["Close"].shift(1).rolling(5).mean()
df["10_DAY_STD"] = df["Close"].shift(1).rolling(10).std()
df["5_DAY_STD"] = df["Close"].shift(1).rolling(5).std()
df["50_DAY_STD"] = df["Close"].shift(1).rolling(50).std()

# df["Close"] = pd.to_numeric(df["Close"])
window_length = 14  # Standard RSI period

# Calculate daily price changes
delta = df['Close'].diff()

# Calculate gains and losses
gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()

# Calculate RS (Relative Strength) and RSI
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# df.set_index("Date", inplace=True)
print(df.head())



df.dropna(inplace=True)
print(df.head())

def get_relevant_data(dff):
    input_df = dff[['MA_10', 'MA_50', 'EMA_10', 'EMA_50', 'RSI', 'MA_5', 'EMA_5', "10_DAY_STD", "5_DAY_STD", "50_DAY_STD"]]
    output_df = dff['Close']
    return input_df, output_df

def split_df(input_df, output_df):
    batch_percentage = 0.90
    input_df_train = input_df.head((int(len(input_df) * batch_percentage)))
    output_df_train = output_df.head((int(len(output_df) * batch_percentage)))
    input_df_test = input_df.tail((len(input_df)-int(len(input_df) * batch_percentage)))
    output_df_test = output_df.tail((len(output_df)-int(len(output_df) * batch_percentage)))
    return input_df_train, output_df_train, input_df_test, output_df_test

def getTestSq3d(x, sequence_length):
    result = []
    for index in range(len(x) - sequence_length):
        result.append(x[index: index + sequence_length])
    return np.array(result)

def getTestSq2d(y, sequence_length):
    result = []
    for index in range(len(y) - sequence_length):
        result.append(y[index + sequence_length])
    return np.array(result)


def get_tensor(x):
    return torch.tensor(x, dtype=torch.float32)



def get_model():
    return StockLSTM(10, 15, 1)

def train_model(model, input_tensor, target_tensor):

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        output= model(input_tensor)

        losses = loss(output, target_tensor)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(num_epochs):
        model.train()
        output= model(input_tensor)

        losses = loss(output, target_tensor)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()





def model_eval(model, input_tensor, actual_tensor):
    y_pred = model(input_tensor)
    loss = torch.nn.MSELoss()
    print(f"Loss: {loss(y_pred, actual_tensor).item()}")

def plot_predictions(model, input_tensor, actual_tensor, actual_dates, scaler2):
    y_pred = model(input_tensor)
    y_pred_unscaled = y_pred.detach().numpy()
    actual_unscaled = actual_tensor.detach().numpy()
    y_pred_unscaled = scaler2.inverse_transform(y_pred_unscaled)
    actual_unscaled = scaler2.inverse_transform(actual_unscaled)
    plt.plot(actual_unscaled, label="Actual")
    plt.plot(y_pred_unscaled, label="Predicted")
    plt.legend()
    plt.show()

def predict_next_day(model):
    pass


X, y = get_relevant_data(df)
y = pd.DataFrame(y)
X_train,  y_train, X_test, y_test = split_df(X,y)
print(X_train.head())
print(y_train.head())
Scaler1 = MinMaxScaler()
X_train_scaled = Scaler1.fit_transform(X_train)
X_test_scaled = Scaler1.transform(X_test)
Scaler2 = MinMaxScaler()
y_train_scaled = Scaler2.fit_transform(y_train)
y_test_scaled = Scaler2.transform(y_test)

seq_length = 5

X_train_sq3d = getTestSq3d(X_train_scaled, seq_length)
X_test_sq3d = getTestSq3d(X_test_scaled, seq_length)
y_train_sq2d = getTestSq2d(y_train_scaled, seq_length)
y_test_sq2d = getTestSq2d(y_test_scaled, seq_length)




X_train_tensor = get_tensor(X_train_sq3d)
X_test_tensor = get_tensor(X_test_sq3d)
y_train_tensor = get_tensor(y_train_sq2d)
y_test_tensor = get_tensor(y_test_sq2d)

model = get_model()
train_model(model, X_train_tensor, y_train_tensor)

plot_predictions(model, X_train_tensor, y_train_tensor, y_train_sq2d, Scaler2)
plot_predictions(model, X_test_tensor, y_test_tensor, y_test_sq2d, Scaler2)

print(model_eval(model, X_test_tensor, y_test_tensor))

df["MA_5"] = pd.to_numeric(df["MA_5"])
df["EMA_10"] = pd.to_numeric(df["EMA_10"])
df["EMA_5"] = pd.to_numeric(df["EMA_5"])
plt.plot(df.tail(len(df)-(int(.9*len(df))))["EMA_5"], label="EMA_5")
plt.plot(y_test, label="Close")
plt.legend()
plt.show()
