from keras.layers import Embedding, Input, Dropout, Dense,\
    Concatenate, Add, GRU, MaxPool1D, AveragePooling1D, Flatten
from keras.layers.core import Activation
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras import regularizers, optimizers
from keras.models import Model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from keras.utils import plot_model
import tensorflow as tf
from keras import backend as K

# use fscore as eval metric
def fscore(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# model structure
def creat_model(gpu,
                model_name,
                mode=None,
                nb_class=24,
                nb_filter=[128, 256, 128],
                wsize=[1, 2, 3],
                maxlen=20,
                cvsize=5000,
                wvsize=5000,
                denseuni=512,
                cnn_dropout=0.2,
                l2=0.001,
                lr=0.01,
                w2v=False,
                w2vdim=128,
                w2vmat=None):

    if mode == 'char_word':
        input_char = Input(shape=(maxlen,), dtype='int32')
        input_word = Input(shape=(maxlen,), dtype='int32')

        if w2v:
            embedding_c = Embedding(input_dim=cvsize + 1, output_dim=w2vdim, weights=[w2vmat],
                                    input_length=maxlen, trainable=True)(input_char)
            embedding_w = Embedding(input_dim=wvsize + 1, output_dim=w2vdim, weights=[w2vmat],
                                    input_length=maxlen, trainable=True)(input_word)
        else:
            embedding_c = Embedding(input_dim=cvsize + 1, output_dim=w2vdim, input_length=maxlen)(input_char)
            embedding_w = Embedding(input_dim=wvsize + 1, output_dim=w2vdim, input_length=maxlen)(input_word)

        embedding = Concatenate()([embedding_c, embedding_w])

    else:
        inputs = Input(shape=(maxlen,), dtype='int32')
        vsize = cvsize if mode == 'char' else wvsize

        if w2v:
            embedding = Embedding(input_dim=vsize + 1, output_dim=w2vdim, weights=[w2vmat],
                                  input_length=maxlen, trainable=True)(inputs)
        else:
            embedding = Embedding(input_dim=vsize + 1, output_dim=w2vdim, input_length=maxlen)(inputs)

    conv_1 = []
    for i in range(len(nb_filter)):
        conv = Conv1D(nb_filter[i], padding='same', kernel_initializer="normal",
                      kernel_size=wsize[i], activation='relu')(embedding)
        conv_1.append(conv)

    concat_1 = Concatenate()(conv_1)

    bn_1 = BatchNormalization()(concat_1)

    act_1 = Activation('relu', name='Relu_1')(bn_1)

    conv_2 = []
    for i in range(len(nb_filter)):
        conv = Conv1D(nb_filter[i], padding='same', kernel_initializer="normal",
                      kernel_size=wsize[i], activation='relu')(bn_1)
        conv_2.append(conv)

    concat_2 = Concatenate()(conv_2)

    bn_2 = BatchNormalization()(concat_2)

    add = Add()([bn_1, bn_2])

    gru_l2r = GRU(units=int(denseuni / 2), name='GRU_L')(add)
    gru_r2l = GRU(units=int(denseuni / 2), go_backwards=True, name='GRU_R')(add)
    gru_bi = concatenate([gru_l2r, gru_r2l], axis=1, name='GRU_Bi')

    dropout_1 = Dropout(cnn_dropout)(gru_bi)

    dense = Dense(units=denseuni, activation='relu',
                  kernel_regularizer=regularizers.l2(l2), name='Dense')(dropout_1)

    dropout_2 = Dropout(cnn_dropout)(dense)

    outputs = Dense(units=nb_class, activation='softmax',
                    kernel_regularizer=regularizers.l2(l2), name='Softmax')(dropout_2)
    if mode == 'char_word':
        model = Model(inputs=[input_char, input_word], outputs=outputs, name=model_name)
    else:
        model = Model(inputs=inputs, outputs=outputs, name=model_name)
    adam = optimizers.Adam(lr=lr)

    if gpu:
        from keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=gpu)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[fscore])

    return model
