#-*- encoding: utf-8 -*-

from dataset_keras import dataset

import numpy as np
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Reshape, Lambda
from keras.layers.recurrent import GRU
from keras.layers.merge import add, concatenate
from keras.activations import relu
from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K

import tensorflow as tf

# import sys

# sys.path.append('../')
# from train_keras import graph

(img_h,img_w) = (64,300) 
input_shape = (img_h, img_w,1)
act = 'relu'
conv_filters = [16, 32, 64, 128, 128, 128, 256, 256, 1]
kernel_size = [3, 3, 3, 3, 3, 3, 2, 2]
pool_size = 2
batch_normal = [True]*8
max_pool_2d = [True, True, False, True, False, True, False, False]
max_pool_size = [2,2,0,2,0,2,0,2]
max_pool_stride = [2,2,0,2,0,(1,2),0,(1,2)]

time_dense_size = 32
rnn_size = 128

def convRelu(input_data, i, batch_normal=False, max_pool=False):
    
    input_data = Conv2D(conv_filters[i],kernel_size[i],
                  padding='same', activation=relu, 
                  kernel_initializer='he_normal', 
                  name='conv{}'.format(i+1))(input_data)
    if batch_normal:
        input_data = BatchNormalization(axis=-1)(input_data)
    if max_pool:
        input_data = MaxPooling2D(max_pool_size[i], max_pool_stride[i], 
                                  padding='same')(input_data)
    return input_data

input_data = Input(name='the_input', shape=input_shape, dtype='float32')

inner = input_data
for index,_ in enumerate(kernel_size):
    inner = convRelu(inner, index, batch_normal=batch_normal[index], 
                     max_pool=max_pool_2d[index])

print(np.shape(inner))    
conv_to_rnn_dims = (19, 8*256)
inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal',
           name='gru1')(inner)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True,
             kernel_initializer='he_normal', name='gru1_b')(inner)

gru1_merged = add([gru_1, gru_1b])

gru_2 = GRU(rnn_size, return_sequences=True, 
           kernel_initializer='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, 
             kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

inner = Dense(17, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
y_pred = Activation('softmax', name='softmax')(inner)


base_model = Model(input=input_data, output=y_pred)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc_decode(y_pred):
    # y_pred = base_model.predict_on_batch(imgs)
    # shape = y_pred[:,2:,:].shape
    # out = K.get_value(K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(50)*17)[0][0])[:, :11]
    out = K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(1)*17)[0][0]
    return out



labels = Input(name='the_labels', shape=[11], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')            ([base_model.output, labels, input_length, label_length])

sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
# graph.append(tf.get_default_graph())

out = Lambda(ctc_decode, output_shape=[None,], name='decoder')(y_pred)
decode = K.function([y_pred],[out])

model_predict = K.function([model.layers[0].input],[model.layers[-5].output])
model_decode = K.function([model.layers[-5].output],[out])


from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import os



batch_size = 50
if __name__ == '__main__':
    data = dataset()
    valid_data = dataset(train=False)
    model, base_model = create_model()
    if os.path.exists("model/model_gtu_best.h5"):
        model.load_weights("model/model_gtu_best.h5")
    history = model.fit_generator(data.get_batch(), steps_per_epoch=int(0.7*100000/batch_size),
                                    epochs=50,
                        validation_data=valid_data.get_batch(),
                                  validation_steps=int(0.1*100000/batch_size),
                       callbacks=[ReduceLROnPlateau('loss'),
                                 ModelCheckpoint('model/model_gru_best.h5',save_best_only=True)])