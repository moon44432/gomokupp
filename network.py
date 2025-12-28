

from keras.layers import Activation, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from hparams import board_width, dn_filters, dn_kernel_size, dn_block_num
import os

DN_INPUT_SHAPE = (board_width, board_width, 2)
DN_OUTPUT_SIZE = board_width ** 2


def conv(filters, kernel_size):
    return Conv2D(filters, kernel_size, padding='same', use_bias=False,
                  kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))


def conv_block(filters, kernel_size):
    def f(x):
        x = conv(filters, kernel_size)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    return f


def dual_network():
    if os.path.exists('./model/best.h5'):
        return

    input = Input(shape=DN_INPUT_SHAPE)

    x = conv_block(dn_filters, dn_kernel_size)(input)
    for i in range(dn_block_num - 1):
        x = conv_block(dn_filters, dn_kernel_size)(x)

    x = GlobalAveragePooling2D()(x)

    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005),
              activation='softmax', name='pi')(x)

    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation('tanh', name='v')(v)

    model = Model(inputs=input, outputs=[p, v])
    model.summary()

    os.makedirs('./model/', exist_ok=True)
    model.save('./model/best.h5')

    K.clear_session()
    del model
