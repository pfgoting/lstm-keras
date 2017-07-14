from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib.pyplot as plt
import numpy
from numpy import concatenate
import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime(x, '%Y-%m')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df = df.drop(0)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

def fit_lstm_stateful(train, batch_size, nb_epoch, neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    loss = []
    val_loss = []
    for i in range(nb_epoch):
        print ("{}/{} epoch".format(i+1,nb_epoch))
        history = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False, validation_split=0.05)
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

def fit_lstm_stateless(train, batch_size, nb_epoch, neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(neurons,input_shape=(1,1)))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1,validation_split=0.05)
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

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

# Update LSTM model
def update_model(model, train, batch_size, updates):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    for i in range(updates):
        model.fit(X, y, nb_epoch=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()

def prepare_data(series, nb_seq):
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    # split data into train and test-sets
    train_size = int(len(supervised_values) * nb_seq)
    test_size = len(supervised_values) - train_size
    train, test = supervised_values[0:train_size,:], supervised_values[train_size:len(supervised_values),:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    return scaler, train_scaled, test_scaled

def prepare_forecast_data(series):
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)


# run a repeated experiment
def experiment_repeating(repeats, series, updates):
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    # split data into train and test-sets
    train, test = supervised_values[0:-12], supervised_values[-12:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    # run experiment
    error_scores = list()
    for r in range(repeats):
        # fit the base model
        lstm_model = fit_lstm(train_scaled, 1, 500, 1)
        # forecast test dataset
        train_copy = numpy.copy(train_scaled)
        predictions = list()
        for i in range(len(test_scaled)):
            # update model
            if i > 0:
                update_model(lstm_model, train_copy, 1, updates)
            # predict
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
            # store forecast
            predictions.append(yhat)
            # add to training set
            train_copy = concatenate((train_copy, test_scaled[i,:].reshape(1, -1)))
        # report performance
        rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
        print('%d) Test RMSE: %.3f' % (r+1, rmse))
        error_scores.append(rmse)
    return error_scores

def forecast_forward(series):

    pass
 
# execute the experiment
def run_exp():
    air = 'international-airline-passengers.csv'
    sh = 'shampoo-sales.csv'
    # load dataset
    series = read_csv(air, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    # experiment
    repeats = 10
    results = DataFrame()
    # run experiment
    updates = 2
    results['results'] = experiment(repeats, series, updates)
    # summarize results
    print(results.describe())
    # save results
    results.to_csv('experiment_update_2.csv', index=False)
 
if __name__ == '__main__':
    # init values
    nb_epoch = 500
    batch_size = 1
    neurons = 4
    nb_seq = 0.67
    nb_steps = 10