import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
import numpy as np
import time

# date-time parsing function for loading the dataset
def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')

# create a df to have a supervised learning problem
def timeseries_to_supervised(data,lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1,lag+1)]
    columns.append(df)
    df = pd.concat(columns,axis=1)
    df.fillna(0,inplace=True)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval,len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

# transform series into train and test sets for supervised learning
def prepare_data(series, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = timeseries_to_supervised(scaled_values, 1)
    supervised_values = supervised.values
    # split data into train and test
    train_size = int(len(supervised_values) * n_seq)
    test_size = len(supervised_values) - train_size
    train, test = supervised_values[0:train_size,:], supervised_values[train_size:len(supervised_values),:]
    return scaler, train, test

# fit an LSTM network to training data
def fit_lstm1(train, n_batch, nb_epoch, n_neurons, n_lag):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    loss = []
    val_loss = []
    for i in range(nb_epoch):
        print "{}/{} epoch".format(i+1,nb_epoch)
        history = model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False, validation_split=0.05)
        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        model.reset_states()

    # plot loss of model
    plt.figure()
    plt.plot(loss,label='loss')
    plt.plot(val_loss,label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('model loss')
    plt.show()
    return model

def fit_lstm2(train, n_batch, nb_epoch, n_neurons, n_lag):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons,input_shape=(1,1)))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(X, y, epochs=100, batch_size=32, verbose=1,validation_split=0.05)
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.title('model loss stateless')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    return model

def fit_lstm(train, n_batch, nb_epoch, n_neurons, n_lag):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()

    model.add(LSTM(
        50,
        input_dim=1,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    history = model.fit(X, y, epochs=100, batch_size=32, verbose=1,validation_split=0.05)
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.title('model loss stateless')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    return model

# make one forecast
def forecast_lstm(model, test, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X, y = test[:,0], test[:,n_lag]
    X = X.reshape(len(X),1,1)
    predictions = model.predict(X,batch_size=n_batch)
    return predictions

def forecast_lstm_sliding(model, test, n_batch, n_steps):
    # reshape input pattern to [samples, timesteps, features]
    X, y = test[:,0], test[:,n_lag]
    predicted = X
    for step in range(n_steps):
        predictions = model.predict(predicted.reshape(len(predicted),1,1),batch_size=1)
        predicted = np.append(predicted, predictions[-1])
    return predicted.reshape(len(predicted),1)

# inverse functions
# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted

# invert data transforms
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = np.array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        # print i
        # print index
        last_ob = series.values[index]
        # print last_ob
        # print '\n'
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    # for i in range(n_seq):
    #     actual = [test[i][0]]
    #     predicted = [forecasts[i][0]]
    actual = [test[-1][0]]
    predicted = [test[-1][0]]
    rmse = sqrt(mean_squared_error(actual, predicted))
    print('t+%d RMSE: %f' % (len(test), rmse))

# plotting
def plot_forecasts(series,actual,forecasts):
    plt.figure()
    plt.plot(series.values[-len(actual):],label='original data')
    plt.plot(actual,label='test data')
    plt.plot(forecasts,label='predicted data')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # init values
    n_epochs = 2
    n_batch = 1
    n_neurons = 4
    n_lag = 1 # columns
    n_seq = 0.67 # no of test vals
    n_steps = 10
    # load data
    # series = pd.read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    series = pd.read_csv('sp500.csv',squeeze=True)

    # prepare data
    scaler, train, test = prepare_data(series,n_seq)
    # fit model
    model = fit_lstm(train,n_batch,n_epochs,n_neurons,n_lag)
    # forecast forward
    forecasts = forecast_lstm(model, test, n_batch)
    # forecast sliding
    # forecasts = forecast_lstm_sliding(model, test, n_batch, n_steps)
    # inverse transform forecasts and test
    forecasts = inverse_transform(series,forecasts,scaler,len(test))
    actual = [row[n_lag:] for row in test]
    actual = inverse_transform(series,actual,scaler,len(test))
    # evaluate forecasts
    evaluate_forecasts(actual, forecasts, n_lag, len(test))
    print series.values[-1]
    print actual[-1]
    print forecasts[-1]
    # plot forecasts
    plot_forecasts(series,actual,forecasts)