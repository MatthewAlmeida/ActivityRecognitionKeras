from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def HARModel(
            batch_size = 128,
            OneD_kernel_size = 8,
            pooling="max",
            dense_size=1000,
            n_filters = 100,
            kernel_reg_param = 0.00005,
            input_shape=(128,6),
            classes=6,
            dpt=0.5,
            ):
    """
    This function returns the keras model object with the architecture
    as detailed in the reference in the readme file, available at
    https://www.sciencedirect.com/science/article/pii/S0957417416302056.

    For this experiment, we vary the input shape parameter as necessary 
    to test the various feature set choices; keras automatically determines
    the layer sizes.
    """
    
    model = Sequential()
    model.add(layers.Conv1D(filters=n_filters, kernel_size=OneD_kernel_size, input_shape = input_shape, kernel_regularizer=regularizers.l2(kernel_reg_param)))
    model.add(layers.BatchNormalization(epsilon=1.001e-5))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(filters=n_filters, kernel_size=OneD_kernel_size, kernel_regularizer=regularizers.l2(kernel_reg_param)))
    model.add(layers.BatchNormalization(epsilon=1.001e-5))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(filters=n_filters, kernel_size=OneD_kernel_size, kernel_regularizer=regularizers.l2(kernel_reg_param)))
    model.add(layers.BatchNormalization(epsilon=1.001e-5))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dropout(dpt))
    model.add(layers.Dense(dense_size, kernel_regularizer=regularizers.l2(kernel_reg_param)))
    model.add(layers.Dense(classes, activation="softmax"))

    return model