from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

SEED=1234
import numpy as np
import math
from matplotlib import pyplot as plt
np.random.seed(SEED)
import keras
from keras import backend as K
import tensorflow as tf
import os, shutil, scipy.io
from model import *

# configure args
tf.set_random_seed(SEED)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
K.set_learning_phase(1)

# data loading part
caseNo = 118
weight_4_mag = 100
weight_4_ang = 1#2*math.pi/360

psse_data = scipy.io.loadmat('dist2_118FASE_data.mat')
print(psse_data['inputs'].shape, psse_data['labels'].shape)

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

print(train_x.shape, val_x.shape)
#Train the model
input_shape = (train_x.shape[1],)

epoch_num = 200
psse_model = nn1_8H_psse(input_shape, train_y.shape[1])
psse_model.fit(train_x, train_y, epochs=epoch_num, batch_size=64)

save_file = '_'.join([str(caseNo), 'nn1_8H_PSSE',
                      'epoch', str(epoch_num)]) + '.h5'

if not os.path.exists('model_logs'):
    os.makedirs('model_logs')
save_path = os.path.join('model_logs', save_file)
print('\nSaving model weights to {:s}'.format(save_path))
psse_model.save_weights(save_path)


# evaluate the model
K.set_learning_phase(0)
val_predic = psse_model.predict(val_x)
scores = psse_model.evaluate(val_x, val_y)
print("\n%s: %.2f%%" % (psse_model.metrics_names[1], scores[1]*100))


#the self.defined distance metric since, to access the distance between predicted and the true
print(val_y.shape[0])
test_no = 3706
def rmse(val_predic, val_y, voltage_distance = np.zeros((test_no,caseNo)), voltage_norm = np.zeros((test_no,1))):
    for i in range(test_no):
        for j in range(caseNo):
            predic_r, predic_i = (1/weight_4_mag)* val_predic[i, j]*math.cos(val_predic[i, j+caseNo]*2*math.pi/360), (1/weight_4_mag)*val_predic[i,j]*math.sin(val_predic[i, j+caseNo]*2*math.pi/360)
            val_r, val_i = (1/weight_4_mag)*val_y[i,j]*math.cos(val_y[i,j+caseNo]*2*math.pi/360), (1/weight_4_mag)*val_y[i][j]*math.sin(val_y[i][j+caseNo]*2*math.pi/360)
            voltage_distance[i,j] = (predic_r-val_r)**2 + (predic_i-val_i)**2
            #print(i, j, val_predic[i, j], val_predic[i, j+caseNo], val_y[i,j], val_y[i,j+caseNo])
        voltage_norm[i,] = (1/caseNo)*np.sqrt(np.sum(voltage_distance[i,:]))
    return np.mean(voltage_norm) *100
print("\n distance from the true states in terms of \|\|_2: %.4f%%" % rmse(val_predic, val_y))



