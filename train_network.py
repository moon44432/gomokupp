
from keras.callbacks import LearningRateScheduler, LambdaCallback
from keras.models import load_model
from keras import backend as K
from pathlib import Path
from network import DN_INPUT_SHAPE
from hparams import rn_epochs, batch_size
import numpy as np
import pickle


def load_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)


def train_network():
    history = load_data()
    xs, y_policies, y_values = zip(*history)

    a, b, c = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)

    model = load_model('./model/best.h5')
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')

    def step_decay(epoch):
        x = 0.001
        if epoch >= 50: x = 0.0005
        if epoch >= 80: x = 0.00025
        return x

    lr_decay = LearningRateScheduler(step_decay)

    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch, logs:
        print('\rTraining network {}/{}'.format(epoch + 1, rn_epochs), end=''))

    model.fit(xs, [y_policies, y_values], batch_size=batch_size, epochs=rn_epochs,
              verbose=0, callbacks=[lr_decay, print_callback])
    print('')

    model.save('./model/latest.h5')

    # 모델 파기
    K.clear_session()
    del model
