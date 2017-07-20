
# Linear Regression of stock prices for
# predicting future prices based on historical data

# Import Python and SKLearn libraries
#

import pandas as pd
import quandl as Quandl
import math, datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle  # for saving classifier data

from sklearn import preprocessing  # for scaling our data
from sklearn import cross_validation  # for testing
from sklearn import svm  # support vector machines : can be used for regression
from sklearn.linear_model import LinearRegression
from matplotlib import style

# needed for unlimited access to WIKI closing stock data
# password/key has been commented out here
# Quandl.ApiConfig.api_key = 'abc' 

style.use('ggplot')

# Choose a ticker symbol for predictions:
# 
ticker_sym = 'WIKI/GOOGL'

#ticker_sym = 'WIKI/AAPL'
#ticker_sym = 'WIKI/IBM'

df = Quandl.get(ticker_sym)


df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

# Note: HL_PCT
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
#df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0


df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Define out features: would like them to end up be being between: -1 to +1
#
df = df[['Adj. Close', 'HL_PCT','PCT_change', 'Adj. Volume']]

#print(df.head())

forecast_col = 'Adj. Close'

# need to fill in NA and NAN with a value
#
df.fillna(-99999, inplace=True) 

# number of days out to forecast, can change 0.01 depending
# on # of days out to forecast
#
# NOTE: added : len - 80 to match April 11 data of YT tut
#
#
# date_offset can be set to 80 to match original tutorial
# or to 0 for current date:

date_offset = 0
forecast_out = int(math.ceil(0.01* (len(df) - date_offset)))

print("forecast_out: ", forecast_out)

# shift column up
#
df['label'] = df[forecast_col].shift(-forecast_out)

#print(df.head())

#print(df.tail())

# Now define our features and labels
# Features are typically: X
# Labels are typically: y

# Define our features
#
X = np.array(df.drop(['label'],1))

# Now we scale X before feeding it to the classifier
# Note: for RT price of HFT price modeling, skip this time intensive step
#
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]

# Pt5
X = X[:-forecast_out]     # latest 30 days


df.dropna(inplace=True)

# Define our labels
#
y = np.array(df['label'])

# reshift points; to ensure we only have X values when we have y vals
# the next line is not needed since we did a dropna above
# X = X[:-forecast_out+1]

# not needed
# df.dropna(inplace=True)

# length of X and y should be equal, roughly @ of trading 
# days for giver symbol
print("X len: ", len(X), "Y len: ", len(y))

# Now we can create our training and testing sets
#
t_size = 0.2

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=t_size)

# Define our classifier:
# n_jobs is # of threads: -1 means max # of threads
clf = LinearRegression(n_jobs=1)  # run as 10 threads for speed

# Now, let's say we wanted to use support vector regression
# Try SVM:
#clf = svm.SVR()    # SVM is much less accurate than LinearRegression for this data

clf.fit(X_train, y_train)             # fit is synonymous with train


# Use pickle to save training data so you don't have to retrain
# every day / every time you want to predict; save once a month, etc
# Use pickle on DigitalOcean with clustered servers to calc classifier
# then save to file, scale down servers when running predictions +
# saved classifier (pickle) file
#

with open('sent_dex_linearregression', 'wb') as f:
    pickle.dump(clf, f)
          
pickle_in = open('sent_dex_linearregression', 'rb')
                
clf = pickle.load(pickle_in)

    
# Test our classifier
# can actually compute separately: accuracy and confidence
#
accuracy = clf.score(X_test, y_test)  # score is synonomous with test

print("LinearRegression: accuracy: ", accuracy)

# LinearRegression: accuracy shows as: 0.96647283423, so 96% accuracy 
# SVM: accuracy shows as: 0.790893216787
# in predicting the price shifted 1& of the days


# check documentation for each model: ie you can massively
# thread LinearRegression

# Now we need to predict based on X data
# we can pass a single value or an array of values here:
forecast_set = clf.predict(X_lately)

#print("forecast_set:", forecast_set
print("accuracy: ", accuracy)
print("forecast_out: ", forecast_out)

df["Forecast"] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()

one_day = 86400
next_unix = last_unix + one_day

next_date = []

# When predicting: no sense of date:
# ML: X are the features, y is the label (the price)
# Data is not a feature, so X is not ready to plot
# need to populate the dataframe with dates + forecast vals

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print("DF_Head: ", df.head())

print("DF_Tail: ", df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.suptitle(ticker_sym)
plt.show()


# Example output:
#
#
#Output:
#
#forecast_out:  31
#X len:  2974 Y len:  2974
#LinearRegression: accuracy:  0.963489066004
#accuracy:  0.963489066004
#forecast_out:  31
#DF_Head:              Adj. Close    HL_PCT  PCT_change  Adj. Volume   label  Forecast
#2004-08-19      50.170  3.707395    0.340000   44659000.0  67.530       NaN
#2004-08-20      54.155  0.710922    7.227007   22834300.0  69.185       NaN
#2004-08-23      54.700  3.729433   -1.218962   18256100.0  68.540       NaN
#2004-08-24      52.435  6.417469   -5.726357   15247300.0  69.425       NaN
#2004-08-25      53.000  1.886792    0.990854    9188600.0  68.865       NaN
#DF_Tail:              Adj. Close  HL_PCT  PCT_change  Adj. Volume  label    Forecast
#2016-07-25         NaN     NaN         NaN          NaN    NaN  808.721474
#2016-07-26         NaN     NaN         NaN          NaN    NaN  807.827818
#2016-07-27         NaN     NaN         NaN          NaN    NaN  808.761948
#2016-07-28         NaN     NaN         NaN          NaN    NaN  810.877601
#2016-07-29         NaN     NaN         NaN          NaN    NaN  810.258327


