#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import time
import math
import tensorflow as tf
import keras
from keras import optimizers, Sequential
from keras.models import Model
#from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
from numpy import arange, sin, pi, random
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import scipy.integrate as integrate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr

np.random.seed(1234)  
PYTHONHASHSEED = 0

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import Data
# 

# In[ ]:


data = np.load('/content/drive/MyDrive/Forecasting_Anomaly_Detection_Auto_LSTM-master/Anomaly_detection/scada1_1.npz')
lst = data.files
# for item in lst:
#     print(item)
#     print(data[item])
X_train = data['training']
X_test = data['test']
idx_anomaly_test = data['idx_anomaly_test']
normal_test_data = X_test[~idx_anomaly_test]
anomalous_test_data = X_test[idx_anomaly_test]
test_index = data['idx_anomaly_test']
#print(X_test.shape)
y_test = [1]*len(data['t_test'])
for i in test_index:
  y_test[i-1] = -1
#print(y_test)
#Split X_train and X_test according to the ratio of 80:20
X_train = X_train[0:40192]
X_test = X_test[0:10048]
y_test = y_test[0:10048]
#Reshape the input data to fit LSTM model
X_train = np.reshape(X_train, (2512,16,16))
X_test = np.reshape(X_test, (628,16,16))
sequence_length =16
DATA_SPLIT_PCT = 0.2
SEED = 123 #used to help randomly select the data points
batch_size = 50
epochs = 30
X_train_X, X_valid = train_test_split(X_train, test_size=DATA_SPLIT_PCT, random_state=SEED)
timesteps =  16 # equal to the sequence_length
n_features =  16
print(X_train_X.shape)
print(X_valid.shape) 


# #LSTM Model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'results = {}\nfor num_cells in [64]:\n    for lr in [1e-3]:\n            print(\'Running with\', num_cells, \n                  \'LSTM cells\'\n                  \'and learning rate =\', lr, \'...\')\n\n            # build network\n            lstm_autoencoder = Sequential()\n            # Encoder\n            lstm_autoencoder.add(LSTM(100, activation=\'relu\', input_shape=(timesteps, n_features), return_sequences=True))\n            lstm_autoencoder.add(LSTM(25, activation=\'relu\', return_sequences=False))\n            lstm_autoencoder.add(RepeatVector(timesteps))\n            # Decoder\n            lstm_autoencoder.add(LSTM(25, activation=\'relu\', return_sequences=True))\n            lstm_autoencoder.add(LSTM(100, activation=\'relu\', return_sequences=True)) \n            lstm_autoencoder.add(TimeDistributed(Dense(n_features)))\n            # lstm_autoencoder.add(tf.keras.layers.Flatten())\n            lstm_autoencoder.summary()\n            adam = tf.optimizers.Adam(lr)\n            lstm_autoencoder.compile(loss=\'mse\', optimizer=adam)\n            cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",\n                               save_best_only=True,\n                               verbose=0)\n            tb = TensorBoard(log_dir=\'./logs\',\n                histogram_freq=0,\n                write_graph=True,\n                write_images=True)\n            lstm_autoencoder_history = lstm_autoencoder.fit(X_train, X_train, \n                                                epochs=epochs, \n                                                batch_size=batch_size, \n                                                validation_data=(X_valid, X_valid),\n                                                verbose=2, callbacks = [keras.callbacks.EarlyStopping(monitor=\'val_loss\', min_delta=0, patience=10, verbose=0, mode=\'auto\')]).history\n            print("Predicting...")\n            predicted_train = lstm_autoencoder.predict(X_train)\n            predicted = lstm_autoencoder.predict(X_test)         \n')


# #Anomaly Detection using OCSVM

# In[ ]:


e = X_train - predicted_train
nsamples, nx, ny = e.shape
d2_e = e.reshape((nsamples*nx,ny))
X_train1 = d2_e
X_test1 = X_test.reshape(10048,16)
from sklearn.svm import OneClassSVM
model = OneClassSVM(kernel = 'rbf', gamma = 0.07, nu = 0.003).fit(X_train1)
# model = OneClassSVM(kernel = 'rbf', gamma = 0.008, nu = 0.001).fit(X_train1)
##{'gamma': 0.001, 'kernel': 'rbf', 'nu': 0.001}
y_predict = model.predict(X_test1)
precision = precision_score(y_test, y_predict)
recall    = recall_score(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)
print ('Accuracy : ', accuracy)
print ('Precision : ', precision)
print ('Recall: ', recall)
print ('F1_score: ', f1)


# In[ ]:


# for i in range(1,10):
#   for j in range(1,10):
#     model = OneClassSVM(kernel = 'rbf', gamma = 0.01*i, nu = 0.001*j).fit(X_train1)
#     y_predict = model.predict(X_test1)
#     precision = precision_score(y_test, y_predict)
#     recall    = recall_score(y_test, y_predict)
#     f1 = f1_score(y_test, y_predict)
#     print(i)
#     print(j)
#     print ('Precision : ', precision)
#     print ('Recall: ', recall)
#     print ('F1_score: ', f1)
#7/3
#Precision :  0.8470223599734337
#Recall:  0.9627579265223956
#F1_score:  0.9011894947591567


# In[ ]:


from sklearn.metrics import PrecisionRecallDisplay
y_score = model.decision_function(X_test1)
display = PrecisionRecallDisplay.from_predictions(y_test, y_score, name="LinearSVC")
_ = display.ax_.set_title("2-class Precision-Recall curve")


# In[ ]:


# for i in range(1,10):
#   for j in range(1,9):
#     model = OneClassSVM(kernel = 'rbf',degree = j, gamma = 0.001*i, nu = 0.002*i).fit(X_train1)
#     y_predict = model.predict(X_test1)
#     f1 = f1_score(y_test, y_predict)
#     print ('F1_score: ', f1)


# In[ ]:


# predicted_train = lstm_autoencoder.predict(X_train)
# predicted = lstm_autoencoder.predict(X_test)
# mse = np.mean(np.power(flatten(X_test) - flatten(predicted), 2), axis=1)
# mse_train = np.mean(np.power(flatten(X_train) - flatten(predicted_train), 2), axis=1)
# params = {'bandwidth': np.linspace(0, 0.5, 10)}
# grid = GridSearchCV(KernelDensity(), params, cv = 20)
# grid.fit(mse_train[:, None])
# h=grid.best_estimator_.bandwidth2
# tau=FindThreshold(mse_train,h,0.56)


# #Model Explantation 

# In[ ]:


lstm_autoencoder1 = Sequential()
# Encoder
lstm_autoencoder1.add(LSTM(100, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
lstm_autoencoder1.add(LSTM(25, activation='relu', return_sequences=False))
lstm_autoencoder1.add(RepeatVector(timesteps))
# Decoder
lstm_autoencoder1.add(LSTM(25, activation='relu', return_sequences=True))
lstm_autoencoder1.add(LSTM(100, activation='relu', return_sequences=True))
lstm_autoencoder1.add(TimeDistributed(Dense(n_features)))
lstm_autoencoder1.add(tf.keras.layers.Flatten())
lstm_autoencoder1.summary()


# In[ ]:


#transfer model
lstm_autoencoder1.layers[0].set_weights(lstm_autoencoder.layers[0].get_weights())
lstm_autoencoder1.layers[1].set_weights(lstm_autoencoder.layers[1].get_weights())
lstm_autoencoder1.layers[2].set_weights(lstm_autoencoder.layers[2].get_weights())
lstm_autoencoder1.layers[3].set_weights(lstm_autoencoder.layers[3].get_weights())
lstm_autoencoder1.layers[4].set_weights(lstm_autoencoder.layers[4].get_weights())
lstm_autoencoder1.layers[5].set_weights(lstm_autoencoder.layers[5].get_weights())
predicted_train1 = lstm_autoencoder1.predict(X_train)
# print(predicted_train1)


# In[ ]:


get_ipython().system('pip  install shap')


# In[ ]:


import shap 
# X_train2 = X_train[0:400]
X_train2 = X_train[:]
explainer = shap.GradientExplainer(lstm_autoencoder1, X_train2)
X_test_2 = X_test[0:16]
shap_values  = explainer.shap_values(X_test_2)


# In[ ]:


shap.initjs()


# In[ ]:


import pandas as pd
features = ['address','function','length','setpoint','gain','reset rate','deadband','cycle time','rate','system mode','control scheme','pump','solenoid','pressure measurement','crc rate','command reponse']
shap_values_2D = shap_values[0].reshape(-1,16)
X_test_2D = X_test_2.reshape(-1,16)
x_test_2d = pd.DataFrame(data=X_test_2D, columns = features)
# x_test_2d.corr()
shap.summary_plot(shap_values_2D, x_test_2d)

