import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

class MomentumScheduler(Callback):
    """
    This class is a modification of Keras' built-in
    Learning rate scheduler to support the scheduling
    of the SGD momentum along with the learning
    rate. 
    """

    def __init__(self, schedule, verbose=0):
        super(MomentumScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'momentum'):
            raise ValueError('Optimizer must have a "momentum" attribute.')
        momentum = float(K.get_value(self.model.optimizer.momentum))
        try:  # new API
            momentum = self.schedule(epoch, momentum)
        except TypeError:  # old API for backward compatibility
            momentum = self.schedule(epoch)
        if not isinstance(momentum, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                            'should be float.')
        K.set_value(self.model.optimizer.momentum, momentum)
        if self.verbose > 0:
            print('\nEpoch %05d: Momentum scheduler setting momentum '
                'to %s.' % (epoch + 1, momentum))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['momentum'] = K.get_value(self.model.optimizer.momentum)