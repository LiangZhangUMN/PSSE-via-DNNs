'''
 * @author [Liang Zhang]
 * @email [zhan3523@umn.edu]
Different NN models for PSSE provided in this file
'''
import tensorflow as tf

from keras import optimizers
from keras import regularizers

from keras.models import Model
from keras.layers import Dense, Activation, add, Dropout, Lambda
from keras.layers import Input, average
from keras import backend as K
from keras.layers.normalization import BatchNormalization


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta

    squared_loss = 0.5 * K.square(error)
    linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)

def huber_loss_mean(y_true, y_pred):
    return K.mean(huber_loss(y_true, y_pred))

def st_activation(tensor, th = 0.2):
    '''Performs the soft thresholding operation, an alternative activation'''
    cond  = K.abs(tensor) < th
    st_tensor = tensor - th*K.sign(tensor)
    return  tf.where(cond, tf.zeros(tf.shape(tensor)), st_tensor)


def nn1_psse(input_shape, num_classes, weights=None):
    '''
    :param input_shape:
    :param num_classes:
    :param weights: 6 layers
    :return: estimated voltages
    '''
    data = Input(shape=input_shape, dtype='float', name='data')
    dense1 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(data)
    dense2 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense1)
    dense3 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense2)
    dense4 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense3)
    dense5 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense4)

    dense6 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense5)
    predictions = Dense(units = num_classes, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense6)

    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.adam(lr=0.001)

#    sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss=huber_loss_mean,
                  metrics=['mae'])

    return model


def nn1_8H_psse(input_shape, num_classes, weights=None):
    '''
    :param input_shape:
    :param num_classes:
    :param weights: 8 layers
    :return: estimated voltages
    '''
    data = Input(shape=input_shape, dtype='float', name='data')
    dense1 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(data)
    dense2 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense1)
    dense3 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense2)
    dense4 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense3)
    dense5 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense4)

    dense6 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense5)
    dense7 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense6)
    dense8 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense7)
    predictions = Dense(units = num_classes, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense8)

    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.adam(lr=0.001)

#    sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss=huber_loss_mean,
                  metrics=['mae'])

    return model



def lav_psse(input_shape, num_classes, weights=None):
    '''
    :param input_shape:
    :param num_classes:
    :param weights:
    :return: estimated voltages
    '''
    data = Input(shape=input_shape, dtype='float', name='data')
    merged1 = Dense(units = input_shape[0], activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(data)
    u01 = Activation('relu')(merged1)
    dense1 = Dense(units = input_shape[0], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(u01)
    add1 = add([merged1, dense1])
    u02 = Activation('relu')(add1)
    dense2 = Dense(units = input_shape[0], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(u02)
    add2 = add([merged1, dense2])
    u03 = Activation('relu')(add2)

    dense3 = Dense(units = input_shape[0], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(u03)
    dense4 = Dense(units = input_shape[0], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(data)
    merged2 = add([dense3, dense4])
    u11 = Activation('relu')(merged2)
    dense5 = Dense(units = input_shape[0], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(u11)
    add3 = add([merged2, dense5])
    u12 = Activation('relu')(add3)
    dense6 = Dense(units = input_shape[0], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(u12)
    add4 = add([merged2, dense6])
    u13 =  Activation('relu')(add4)

    dense_o1 = Dense(units = input_shape[0], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(data)
    add_o1 = add([u13, dense_o1])
    predictions = Dense(units = num_classes, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(add_o1)

    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.adam(lr=0.001)

    model.compile(optimizer=sgd, loss=huber_loss_mean,
                  metrics=['mae'])

    return model


def st_lav_psse(input_shape, num_classes, weights=None):
    '''
    soft_max activation
    :param input_shape:
    :param num_classes:
    :param weights:
    :return: estimated voltages
    '''
    data = Input(shape=input_shape, dtype='float', name='data')
    merged1 = Dense(units = input_shape[0], activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(data)
    u01 = Lambda(st_activation, name='st0')(merged1)
    dense1 = Dense(units = input_shape[0], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(u01)
    add1 = add([merged1, dense1])
    u02 = Lambda(st_activation, name='st1')(add1)
    dense2 = Dense(units = input_shape[0], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(u02)
    add2 = add([merged1, dense2])
    u03 = Lambda(st_activation, name='st2')(add2)
    dense_o1 = Dense(units = input_shape[0], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(data)
    add_o1 = add([u03, dense_o1])
    predictions = Dense(units = num_classes, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(add_o1)

    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.adam(lr=0.001)

    model.compile(optimizer=sgd, loss=huber_loss_mean,
                  metrics=['mae'])
    return model



