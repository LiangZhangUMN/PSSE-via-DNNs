from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras import optimizers
from keras.utils import np_utils
import os, shutil, scipy.io
from matplotlib import pyplot as plt

from keras import backend as K
import tensorflow as tf
from model import *

# configure args
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
K.set_learning_phase(1)

caseNo = 118
input_dim = caseNo * 2
weight_4_mag = 100.0
weight_4_ang = 1.0#2*math.pi/360

# data loading part
psse_data = scipy.io.loadmat('dist2_118FASE_data.mat')
data_x = psse_data['inputs']
data_y = psse_data['labels']

# scale the mags,
data_y[0:caseNo,:] = weight_4_mag*data_y[0:caseNo,:]
data_y[caseNo:,:] = weight_4_ang*data_y[caseNo:,:]
# seperate them into training 80%, test 20%
split_train = int(0.8*psse_data['inputs'].shape[1])
split_val = psse_data['inputs'].shape[1] - split_train #int(0.25*psse_data['inputs'].shape[1])
train_x = np.transpose(data_x[:, :split_train])
train_y = np.transpose(data_y[:, :split_train])
val_x   = np.transpose(data_x[:, split_train:split_train+split_val])
val_y   = np.transpose(data_y[:, split_train:split_train+split_val])
test_x  = np.transpose(data_x[:, split_train+split_val:])
test_y  = np.transpose(data_y[:, split_train+split_val:])


total_v = len(train_y)
print(train_y.shape, val_y.shape, val_y[0].shape)
#How many timesteps e.g how many characters we want to process in one go
time_steps = 10

#Since our timestep sequence represetns a process for every 100 chars we omit
#the first 100 chars so the loop runs a 100 less or there will be index out of
#range
counter = total_v - time_steps

#Inpput data
vX = []
#output data
y = []
#This loops through all the characters in the data skipping the first 100
for i in range(counter):
    #This one goes from 0-100 so it gets 100 values starting from 0 and stops
    #just before the 100th value
    theInput = train_y[i:i+time_steps]
    #With no : you start with 0, and so you get the actual 100th value
    #Essentially, the output Chars is the next char in line for those 100 chars
    #in X
    theOutput = train_y[i + time_steps]
    #Appends every 100 chars ids as a list into X
    vX.append(theInput)
    #For every 100 values there is one y value which is the output
    y.append(theOutput)

#Len charX represents how many of those time steps we have
#Our features are set to 1 because in the output we are only predicting 1 char
#Finally numberOfCharsToLearn is how many character we process
X = np.reshape(vX, (len(vX), time_steps, input_dim))
y = np.reshape(y, (len(y), len(y[0])))

rnnFaseModel = stack_rnn_fase(X.shape, y.shape[1], weights=None)

rnnFaseModel.fit(X, y, epochs = 200, batch_size=32)
rnnFaseModel.save_weights("RNN4FASE.hdf5")
#model.load_weights("LSTMforStatePredic.hdf5")

#this is for forecasting the new voltage profile
K.set_learning_phase(0)
ans = [] # ans collects the result of all the voltages
test_no = 3706
v_start = vX[-1]
for i in range(test_no):
    x = np.reshape(v_start, (1, time_steps, input_dim))
    pred = rnnFaseModel.predict(x)
    ans.append(pred)
    true_v = np.reshape(val_y[i], pred.shape)
    v_start = np.concatenate((v_start, true_v), axis = 0)
    v_start = v_start[1:]


voltage_distance = np.zeros((test_no,caseNo))
voltage_norm = np.zeros((test_no,1))
val_predic = np.reshape(ans, (test_no, input_dim))


for i in range(test_no):
    for j in range(caseNo):
        predic_r, predic_i = (1/weight_4_mag)* val_predic[i, j]*math.cos(val_predic[i, j+caseNo]*2*math.pi/360), (1/weight_4_mag)*val_predic[i,j]*math.sin(val_predic[i, j+caseNo]*2*math.pi/360)
        val_r, val_i = (1/weight_4_mag)*val_y[i,j]*math.cos(val_y[i,j+caseNo]*2*math.pi/360), (1/weight_4_mag)*val_y[i][j]*math.sin(val_y[i][j+caseNo]*2*math.pi/360)
        voltage_distance[i,j] = (predic_r-val_r)**2 + (predic_i-val_i)**2
        #print(i, j, val_predic[i, j], val_predic[i, j+caseNo], val_y[i,j], val_y[i,j+caseNo])
    voltage_norm[i,] = (1/caseNo)*np.sqrt(np.sum(voltage_distance[i,:]))
print("\n distance from the true states in terms of \|\|_2: %.4f%%" % (np.mean(voltage_norm)*100))
print(voltage_norm)
print(np.amin(voltage_norm), np.argmin(voltage_norm))

#add the plot here

timeslot = 200
points = caseNo

print(val_y[timeslot].shape)
print(val_predic.shape)

plt.subplot(2,1,1)
plt.plot(range(points), (1/weight_4_mag)*val_y[timeslot][0:points], color='r')
plt.hold
plt.plot(range(points), (1/weight_4_mag)*val_predic[timeslot][0:points], color='b')
plt.ylabel('Voltage mag (pu)')
plt.title('Voltages at time  ' + str(timeslot))

plt.subplot(2,1,2)
plt.plot(range(points), val_y[timeslot][caseNo:caseNo+points], color='r')
plt.hold
plt.plot(range(points), val_predic[timeslot][caseNo:caseNo+points], color='b')
plt.ylabel('Voltage ang (pu)')
plt.xlabel('Bus index')

plt.show()






