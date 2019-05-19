'''
 * @author [Liang Zhang]
 * @email [zhan3523@umn.edu]
Different NN models for PSSE provided in this file
'''
import tensorflow as tf

from keras import optimizers
from keras import regularizers

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, add, Dropout, Lambda, LSTM
from keras.layers import Input, average, TimeDistributed, SimpleRNN, LeakyReLU
from keras import backend as K
from keras.layers.normalization import BatchNormalization


def huber_loss(y_true, y_pred, clip_delta=0.010):
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta

    squared_loss = 0.5 * K.square(error)
    linear_loss  = 0* clip_delta * (K.abs(error) - 0 * clip_delta)

    return tf.where(cond, linear_loss, squared_loss)

def huber_loss_mean(y_true, y_pred):
    return K.mean(huber_loss(y_true, y_pred))

def st_activation(tensor, th = 0.2):
    '''Performs the soft thresholding operation'''
    cond  = K.abs(tensor) < th
    st_tensor = tensor - th*K.sign(tensor)
    return  tf.where(cond, tf.zeros(tf.shape(tensor)), st_tensor)


def ANN_fase(input_shape, output_shape, weights = None):
    # input_shape (batch_size, time_series, input_dim)
    # out_shape: dimension of the voltage vector
    input_shape = input_shape[1:]
    data = Input(shape=input_shape, dtype='float', name='data')
    dense1 = Dense(units = 236, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(data)
    predictions = Dense(units = output_shape, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense1)

    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.adam(lr=0.001)

    model.compile(optimizer=sgd, loss='mae',
                  metrics=['mae'])
    return model


def rnn_fase(input_shape, output_shape, weights=None):
    # input_shape (batch_size, time_series, input_dim)
    # out_shape: dimension of the voltage vector
    model = Sequential()
    #Since we know the shape of our Data we can input the timestep and feature data
    #The number of timestep sequence are dealt with in the fit function
    model.add(SimpleRNN(input_shape[2], input_shape=(input_shape[1], input_shape[2])))
    #model.add(Dropout(0.2))
    #number of features on the output
    model.add(Dense(output_shape, activation='linear',   use_bias=True, kernel_regularizer=None))
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.adam(lr=0.001)
    model.compile(optimizer=sgd, loss='mean_squared_error',
                      metrics=['mae'])
    return model



def lstm_fase(input_shape, output_shape, weights=None):
    # input_shape (batch_size, time_series, input_dim)
    # out_shape: dimension of the voltage vector
    model = Sequential()
    #Since we know the shape of our Data we can input the timestep and feature data
    #The number of timestep sequence are dealt with in the fit function
    model.add(LSTM( input_shape[2], input_shape=(input_shape[1], input_shape[2])))
    #model.add(Dropout(0.2))
    #number of features on the output
    model.add(Dense(output_shape, activation='linear', use_bias=True))
    sgd = optimizers.adam(lr=0.001)
    model.compile(optimizer=sgd, loss='mean_squared_error',
                      metrics=['mae'])
    return model


def lstm_dinput_fase(input_shape, output_shape, weights=None):
    # input_shape (batch_size, time_series, input_dim)
    # out_shape: dimension of the voltage vector
    model = Sequential()
    #Since we know the shape of our Data we can input the timestep and feature data
    #The number of timestep sequence are dealt with in the fit function
    model.add(TimeDistributed(Dense(256), input_shape=(input_shape[1], input_shape[2])))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    #number of features on the output
    model.add(Dense(output_shape, activation='linear'))
    sgd = optimizers.adam(lr=0.001)
    model.compile(optimizer=sgd, loss='mean_squared_error',
                      metrics=['mae'])
    return model


def stack_rnn_fase(input_shape, output_shape, weights=None):
    # input_shape (batch_size, time_series, input_dim)
    # out_shape: dimension of the voltage vector

    model = Sequential()
    #Since we know the shape of our Data we can input the timestep and feature data
    #The number of timestep sequence are dealt with in the fit function
    model.add(SimpleRNN(input_shape[2], activation='relu',  return_sequences=True, input_shape=(input_shape[1], input_shape[2])))
    #model.add(Dropout(0.2))
    #number of features on the output
    model.add(SimpleRNN(input_shape[2], activation='relu',  return_sequences=True))
    #model.add(SimpleRNN(input_shape[2], return_sequences=True))
    model.add(SimpleRNN(input_shape[2], activation='relu', ))
    model.add(Dense(output_shape, activation='linear',  use_bias=True, kernel_regularizer=None))
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.adam(lr=0.001)
    model.compile(optimizer=sgd, loss= 'mae',
                      metrics=['mae'])
    return model



def rnn_plnet_fase(input_shape, output_shape, weights=None):
    # input_shape (batch_size, time_series, input_dim)
    # out_shape: dimension of the voltage vector

    inputs = Input(shape=(input_shape[1], input_shape[2]), dtype='float', name='data')

    #Define a plnet
    data = Input(shape=input_shape[2:], dtype='float', name='data')
    merged1 = Dense(units = input_shape[2], activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(data)
    u01 = Activation('relu')(merged1)
    dense1 = Dense(units = input_shape[2], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(u01)
    add1 = add([merged1, dense1])
    u02 = Activation('relu')(add1)
    dense2 = Dense(units = input_shape[2], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(u02)
    add2 = add([merged1, dense2])
    u03 = Activation('relu')(add2)


    dense3 = Dense(units = input_shape[2], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(u03)
    dense4 = Dense(units = input_shape[2], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(data)
    merged2 = add([dense3, dense4])

    u11 = Activation('relu')(merged2)
    dense5 = Dense(units = input_shape[2], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(u11)
    add3 = add([merged2, dense5])
    u12 = Activation('relu')(add3)
    dense6 = Dense(units = input_shape[2], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(u12)
    add4 = add([merged2, dense6])
    u13 =  Activation('relu')(add4)

    dense_o1 = Dense(units = input_shape[2], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(data)
    add_o1 = add([u13, dense_o1])


    #predictions = Dense(units = output_shape, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(add_o1)

    plnet = Model(inputs=data, outputs=add_o1)
    out = TimeDistributed(plnet)(inputs)

    
    rnn1 = SimpleRNN(input_shape[2],  activation='relu', return_sequences=True)(out)
    rnn2 = SimpleRNN(input_shape[2],  activation='relu', return_sequences=True)(rnn1)
    rnn3 = SimpleRNN(input_shape[2],  activation='relu', return_sequences=False)(rnn2)

    new_v = Dense(output_shape, activation='linear')(rnn3)

    model = Model(inputs=inputs, outputs=new_v)
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.adam(lr=0.001)

    model.compile(optimizer=sgd, loss = 'mae',
                  metrics=['mae'])

    return model

def pretrained_rnn_plnet_fase(input_shape, output_shape, weights = None, pl_weights = None):
    # input_shape (batch_size, time_series, input_dim)
    # out_shape: dimension of the voltage vector

    inputs = Input(shape=(input_shape[1], input_shape[2]), dtype='float', name='data')

    #Define a plnet
    learnable = False
    data = Input(shape=input_shape[2:], dtype='float', name='data')
    merged1 = Dense(units = input_shape[2], activation=None, use_bias=True, trainable = learnable, kernel_initializer='glorot_uniform', bias_initializer='zeros')(data)
    u01 = Activation('relu')(merged1)
    dense1 = Dense(units = input_shape[2], activation=None, use_bias=True, trainable = learnable, kernel_initializer='glorot_uniform')(u01)
    add1 = add([merged1, dense1])
    u02 = Activation('relu')(add1)
    dense2 = Dense(units = input_shape[2], activation=None, use_bias=True,  trainable = learnable, kernel_initializer='glorot_uniform')(u02)
    add2 = add([merged1, dense2])
    u03 = Activation('relu')(add2)


    dense3 = Dense(units = input_shape[2], activation=None, use_bias=True, trainable = learnable, kernel_initializer='glorot_uniform')(u03)
    dense4 = Dense(units = input_shape[2], activation=None, use_bias=True, trainable = learnable, kernel_initializer='glorot_uniform')(data)
    merged2 = add([dense3, dense4])

    u11 = Activation('relu')(merged2)
    dense5 = Dense(units = input_shape[2], activation=None, use_bias=True, trainable = learnable, kernel_initializer='glorot_uniform')(u11)
    add3 = add([merged2, dense5])
    u12 = Activation('relu')(add3)
    dense6 = Dense(units = input_shape[2], activation=None, use_bias=True, trainable = learnable, kernel_initializer='glorot_uniform')(u12)
    add4 = add([merged2, dense6])
    u13 =  Activation('relu')(add4)

    dense_o1 = Dense(units = input_shape[2], activation=None, use_bias=True, trainable = learnable, kernel_initializer='glorot_uniform')(data)
    add_o1 = add([u13, dense_o1])


    predictions = Dense(units = output_shape, activation=None, use_bias=True, trainable = learnable, kernel_initializer='glorot_uniform', bias_initializer='zeros')(add_o1)

    plnet = Model(inputs=data, outputs=predictions)
    if pl_weights is not None:
        plnet.load_weights(pl_weights)


    out = TimeDistributed(plnet)(inputs)


    rnn1 = SimpleRNN(input_shape[2],   activation='relu' ,return_sequences=True)(out)
    rnn2 = SimpleRNN(input_shape[2],   activation='relu' , return_sequences=True)(rnn1)
    rnn3 = SimpleRNN(input_shape[2],  activation='relu',  return_sequences=False)(rnn2)

    new_v = Dense(output_shape, activation='linear')(rnn3)

    model = Model(inputs=inputs, outputs=new_v)
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.adam(lr=0.001)

    model.compile(optimizer=sgd, loss = 'mae',
                  metrics=['mae'])

    return model

def simplified_rpln_fase(input_shape, output_shape, weights=None):
    # input_shape (batch_size, time_series, input_dim)
    # out_shape: dimension of the voltage vector

    model = Sequential()
    #Since we know the shape of our Data we can input the timestep and feature data
    #The number of timestep sequence are dealt with in the fit function
    model.add(SimpleRNN(input_shape[2], return_sequences=True, input_shape=(input_shape[1], input_shape[2])))
    #model.add(Dropout(0.2))
    #number of features on the output
    model.add(SimpleRNN(input_shape[2], return_sequences=True))
    #model.add(SimpleRNN(input_shape[2], return_sequences=True))
    model.add(SimpleRNN(input_shape[2]))
    model.add(Dense(output_shape, activation='linear',  use_bias=True, kernel_regularizer=None))
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.adam(lr=0.001)
    model.compile(optimizer=sgd, loss= 'mean_squared_error',
                      metrics=['mae'])
    return model


def lav_psse(input_shape, num_classes, weights=None):
    '''
    :param input_shape:
    :param num_classes:
    :param weights:
    :return: 6 hidden layer NN model with specified training loss
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


#    u13 =  Dropout(rate=0.5, name='drop_u23')(u13)
    dense_o1 = Dense(units = input_shape[0], activation=None, use_bias=True, kernel_initializer='glorot_uniform')(data)
    add_o1 = add([u13, dense_o1])
#   drop_o1 = Dropout(rate=0.5, name='drop1')(add_o1)

    predictions = Dense(units = num_classes, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(add_o1)

    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.adam(lr=0.001)

    model.compile(optimizer=sgd, loss=huber_loss_mean,
                  metrics=['mae'])

    return model

def nn0_psse(input_shape, num_classes, weights=None):
    '''
    :param input_shape:
    :param num_classes:
    :param weights:
    :return: 1 hidden layer NN model with specified training loss
    '''
    data = Input(shape=input_shape, dtype='float', name='data')
    dense1 = Dense(units = input_shape[0], activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(data)
    dense2 = Dense(units = input_shape[0], activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense1)
    dense3 = Dense(units = input_shape[0], activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense2)

    predictions = Dense(units = num_classes, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense3)

    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights)
#    sgd = optimizers.adam(lr=0.0001)

    sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='mean_squared_error',
                  metrics=['mae'])

    return model

def nn1_psse(input_shape, num_classes, weights=None):
    '''
    :param input_shape:
    :param num_classes:
    :param weights: 6 layers
    :return: 1 hidden layer NN model with specified training loss
    '''
    data = Input(shape=input_shape, dtype='float', name='data')
    dense1 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(data)
    dense2 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense1)
    dense3 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense2)
    dense4 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense3)
    dense5 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense4)

    dense6 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense5)
#    dense7 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense6)
#    dense8 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense7)
#    dense9 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense8)
#    dense10 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense9)

#    drop1 = Dropout(rate=0.5, name='drop1')(dense8)
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
    :param weights: 6 layers
    :return: 1 hidden layer NN model with specified training loss
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
#    dense9 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense8)
#    dense10 = Dense(units = input_shape[0], activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense9)

#    drop1 = Dropout(rate=0.5, name='drop1')(dense8)
    predictions = Dense(units = num_classes, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense8)

    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.adam(lr=0.001)

#    sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss=huber_loss_mean,
                  metrics=['mae'])

    return model








