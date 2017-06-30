import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import numpy as np
import time

# date-time parsing function for loading the dataset
def parser(x):
    return pd.datetime.strptime('190' + x, '%Y-%m')

def parser2(x):
    return pd.datetime.strptime(x,'%Y-%m')

# create a df to have a supervised learning problem
def timeseries_to_supervised(data,lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1,lag+1)]
    columns.append(df)
    df = pd.concat(columns,axis=1)
    df.fillna(0,inplace=True)
    return df

# Transform dataset to stationary. This removes the trends from the data that are dependent on time.
# One way of stationarizing a dataset is through data differencing resulting to seeing the changes
# to the observations from one timestep to the next.
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval,len(dataset)):
        value = dataset[i] - dataset[i-interval]
        diff.append(value)
    return pd.Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0,-1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    # LSTM layer expects input to be in a matrix with the dimensions: [samples, time steps, features]
    X,y = train[:,0:-1], train[:,-1]
    X = X.reshape(X.shape[0],1,X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

    loss = []
    acc = []
    val_loss = []
    for i in range(nb_epoch):
        print '{}/{} epoch'.format(i+1,nb_epoch)
        hist = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False, validation_split=0.05)
        loss.append(hist.history['loss'])
        acc.append(hist.history['acc'])
        val_loss.append(hist.history['val_loss'])
        model.reset_states()
    plt.figure()
    plt.plot(loss,'r-',label='loss')
    plt.plot(val_loss,label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.figure()
    # plt.plot(range(nb_epoch),acc,'g-')
    plt.show()
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1,1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

# init time
t0 = time.time()

# input files
shampoo = 'shampoo-sales.csv'
airplane = 'international-airline-passengers.csv'
sp500 = 'sp500.csv'
sin = 'sinwave.csv'
data = np.arange(1,51,.10)

# load dataset
# series = pd.Series(data)
# series = pd.read_csv(airplane, header=0,
                 # parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series = pd.read_csv(sp500,squeeze=True)
# series = pd.read_hdf('cex-data.hdf','cex-1d').closing

# 1. transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# 2. transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test
train_size = int(len(supervised_values) * 0.67)
test_size = len(supervised_values) - train_size
train, test = supervised_values[0:train_size,:], supervised_values[train_size:len(supervised_values),:]

# 3. transform the scale of the data
scaler, train_scaled, test_scaled = scale(train,test)

def repeat_exp(repeats=2):
    # repeat experiment
    repeats = repeats
    error_scores = []
    for r in range(repeats):
        # fit the model
        lstm_model = fit_lstm(train_scaled, 1, 1, 4) # 1 batch, 3000 epoch, 4 neurons
        # forecast the entire training dataset to build up state for forecasting
        train_reshaped = train_scaled[:,0].reshape(len(train_scaled), 1, 1)
        lstm_model.predict(train_reshaped, batch_size=1)

        # walk-forward validation on the test data
        predictions = list()
        for i in range(len(test_scaled)):
            # make one-step forecast
            X, y = test_scaled[i,0:-1], test_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)
            # yhat = y
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
            # store forecast
            predictions.append(yhat)
            expected = raw_values[len(train)+i+1]
            print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

        # report performance
        rmse = sqrt(mean_squared_error(raw_values[-len(test):], predictions))
        print 'Test RMSE: %.3f' % rmse
        error_scores.append(rmse)

    # summarize results
    results = pd.DataFrame()
    results['rmse'] = error_scores
    print results.describe()
    # plot line of observed vs predicted
    results.boxplot()
    plt.show()

# def one_exp():
# fit the model
lstm_model = fit_lstm(train_scaled, 1, 50, 4) # 1 batch, 3000 epoch, 4 neurons
# forecast the entire training dataset to build up state for forecasting
# train_reshaped = train_scaled[:,0].reshape(len(train_scaled), 1, 1)
# lstm_model.predict(train_reshaped, batch_size=1)

# forecast test
# test_reshaped = test_scaled[:,0].reshape(len(test_scaled), 1, 1)
# predictions = lstm_model.predict(test_reshaped,batch_size=1)

# revert predictions
# final_pred = []
# for y in range(len(predictions)):
	# invert scaling
	# yhat = invert_scale(scaler, X, y)


# walk-forward validation on the test data
predictions = list()
p = []
trained = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i,0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    p.append(yhat)
    # yhat = y
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    y = invert_scale(scaler,X,y)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    y = inverse_difference(raw_values, y, len(train_scaled)+1-i)
    # store forecast
    predictions.append(yhat)
    trained.append(y)
    expected = raw_values[len(train)+i+1]
    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[-len(test):], predictions))
print 'Test RMSE: %.3f' % rmse
plt.plot(raw_values[-len(test):],label='raw')
plt.plot(predictions,label='predicted')
plt.plot(trained,label='trained')
plt.legend()
plt.show()
# one_exp()
t1 = time.time()
dt = t1-t0
print "Time to train model: {}".format(dt)


# Persistence model forecast as the baseline forecast. It is good to have a baseline performance
# of the model top which future models can be validated.

# 
# history = list(train)
# predictions = list()

# for i in range(len(test)):
#     # make prediction
#     predictions.append(history[-1])

#     # observation
#     history.append(test[i])

# # report performance
# rmse = sqrt(mean_squared_error(test, predictions))
# print('RMSE: %.3f' % rmse)

# # line plot of observed vs predicted
# plt.plot(test)
# plt.plot(predictions)
# plt.show()


# # transform dataset to supervised learning dataset
# supervised = timeseries_to_supervised(X,1)
# print supervised.head()

# # transform to stationary
# differenced = difference(series,1)
# print differenced.head()

# # invert transform
# inverted = list()
# for i in range(len(differenced)):
#     value = inverse_difference(series,differenced[i],len(series)-i)
#     inverted.append(value)
# inverted = pd.Series(inverted)
# print inverted.head()

# # transform scale
# X = X.reshape(len(X),1)
# scaler = MinMaxScaler(feature_range=(-1,1))
# scaler = scaler.fit(X)
# scaled_X = scaler.transform(X)
# scaled_series = pd.Series(scaled_X[:,0])
# print scaled_series.head()

# # invert scale transform
# inverted_X = scaler.inverse_transform(scaled_X)
# inverted_series = pd.Series(inverted_X[:,0])
# print inverted_series.head()

