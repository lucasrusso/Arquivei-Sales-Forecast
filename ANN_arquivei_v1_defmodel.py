
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data Preprocessing: ------------------------------------------------------------

dataset = pd.read_csv('fechamentos_dia_FULL.csv', sep = "[-,]", engine='python')
# line 297 is the first day of 2016


# shuffle, split, normalize
'''Xy = dataset.iloc[297:, 0:4].values

from sklearn.model_selection import train_test_split
Xy_train, Xy_test = train_test_split(Xy, test_size = 0.1, shuffle=False)

np.random.shuffle(Xy_train)

X_train = Xy_train[:, 0:3]
y_train = Xy_train[:, 3]

X_test = Xy_test[:, 0:3]
y_test = Xy_test[:, 3]

from sklearn.preprocessing import MinMaxScaler
norm_X = MinMaxScaler()
X_train = norm_X.fit_transform(X_train)'''

# normalize, slplit shuffle
Xy = dataset.iloc[297:, 0:4].values

from sklearn.preprocessing import MinMaxScaler
norm_Xy = MinMaxScaler()
Xy_normd = norm_Xy.fit_transform(Xy)

from sklearn.model_selection import train_test_split
Xy_train, Xy_test = train_test_split(Xy_normd, test_size = 0.1, shuffle=False)

np.random.shuffle(Xy_train)

X_train = Xy_train[:782, 0:3]
y_train = Xy_train[:782, 3]

X_test = Xy_test[782:, 0:3]
y_test = Xy_test[782:, 3]

# the ANN: -------------------------------------------------------------------------

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
model = Sequential()

model.add(Dense(
        input_dim = 3, units = 3, kernel_initializer='normal', activation = 'relu'))
model.add(Dense(
        units = 6,  kernel_initializer='normal', activation = 'relu'))
model.add(Dense(
        units = 12,  kernel_initializer='normal', activation = 'relu'))
model.add(Dense(
        units = 24,  kernel_initializer='normal', activation = 'relu'))
model.add(Dense(
        units = 48,  kernel_initializer='normal', activation = 'relu'))
model.add(Dense(
        units = 96,  kernel_initializer='normal', activation = 'relu'))
model.add(Dense(
        units = 48,  kernel_initializer='normal', activation = 'relu'))
model.add(Dense(
        units = 24,  kernel_initializer='normal', activation = 'relu'))
model.add(Dense(
        units = 12,  kernel_initializer='normal', activation = 'relu'))
model.add(Dense(
        units = 6,  kernel_initializer='normal', activation = 'relu'))
model.add(Dense(
        units = 3,  kernel_initializer='normal', activation = 'relu'))
model.add(Dense(
        units = 1, kernel_initializer='normal'))

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 50, nb_epoch = 10000)

#ploting the TRAIN SET RESULTS - OK
plt.plot(np.arange(len(X_train[:,2])), y_train, color='blue')
plt.plot(np.arange(len(X_train[:,2])), model.predict(X_train), color='red')
plt.title('Real vs previsto (TRAIN)')
plt.show()

#ploting the TEST SET RESULTS - NOT WORKING
plt.plot(np.arange(len(X_test[:,2])), y_test, color='blue')
plt.plot(np.arange(len(X_test[:,2])), model.predict(X_test), color='red')
plt.title('Real vs previsto (TEST)')
plt.show()

