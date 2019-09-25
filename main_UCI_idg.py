import os
import numpy as np
import math

import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, Callback

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from ARModel import HARModel
from ARUtils import MomentumScheduler, get_one_cycle_lr_fn, get_one_cycle_momentum_fn

"""
Main training function. Runs three activity recognition models:

1) Using all features (Accelerometer X,Y,Z axes, Gyroscope X,Y,Z axes)
2) Using only Accelerometer (Accelerometer X,Y,Z axes)
3) Using only Gyroscope (Gyroscope X,Y,Z axes)

All results are written to a log file in the Experiment_logs Subdirectory.

"""

# We use an 8-GPU server, and Keras will take every GPU by default
# if we fail to be courteous and tell it to only take one!

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set data and log directory paths
data_dir = "./data/"
experiment_log_dir = "./Experiment_logs/"

# Define label list for log readability
labels = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

if __name__ == "__main__":

    # Define Hyperparams:

    learning_rate = 0.003
    batch_size = 256
    epochs = 100
    dropout = 0.6

    # The accelerometer features are the first three in the third numpy axis,
    # gyroscope are the next three. We define slice objects to select the 
    # features so we can loop through them easily.

    all_features = slice(0,6)
    accelerometer_only = slice(0,3)
    gyroscope_only = slice(3,6)

    feature_sets = {
        "All_Features": all_features, 
        "Accelerometer_Only": accelerometer_only, 
        "Gyroscope_Only": gyroscope_only
        }

    # Begin building the log entry
    
    log_entry = "------------------------------"
    log_entry += "UCI Data Set\n\n"
    log_entry = "Hyperparameters:\n\n"
    log_entry += "LR: {0}, batch_size: {1}, epochs: {2}, dropout: {3}\n\n".format(
        learning_rate,
        batch_size,
        epochs,
        dropout
    )

    # Load the data from the numpy files. We've cleaned this from the 
    # raw data already. Each sample is 128 time steps long. Data was 
    # collected at 30Hz, so our windows are slightly longer than four
    # seconds of activity.

    X_train = np.load(data_dir + "X_train_all_features.npy")
    X_test = np.load(data_dir + "X_test_all_features.npy")
    y_train = np.load(data_dir + "y_train.npy")
    y_test = np.load(data_dir + "y_test.npy")

    # Here we loop through the slices and train a model from scratch on each
    # feature set.

    for feature_set_name, feature_set_slice in feature_sets.items():
        log_entry += "--------------------\n"
        log_entry += "{0} Model:\n\n".format(feature_set_name)

        # Slice overall data
        X_train_fs = X_train[:, :, feature_set_slice]
        X_test_fs = X_test[:, :, feature_set_slice]

        # Calculates the input shape
        input_shape = X_train_fs.shape[1:]

        """
        The below section normalizes the data to its z-score. This code is difficult 
        to read, but what it does is put the whole training set worth of samples
        end-to-end, then normalize each feature down the time axis and reshape the data 
        back into the proper shape for training. 

        We use sklearn's StandardScaler to do this. The .fit() method calculates the 
        statistics based on the training set, then we scale both the training and 
        test sets based on these values, so we don't have any information leakage
        regarding the testing distribution in the scaling statistics.
        """

        X_train_reshaped = np.reshape(X_train_fs, (X_train_fs.shape[0] * X_train_fs.shape[1], X_train_fs.shape[2]))

        scaler = StandardScaler()
        scaler.fit(X_train_reshaped)

        X_train_scaled_not_reshaped = scaler.transform(X_train_reshaped)
        X_train_scaled = np.reshape(X_train_scaled_not_reshaped, X_train_fs.shape)

        X_test_reshaped = np.reshape(X_test_fs, (X_test_fs.shape[0] * X_test_fs.shape[1], X_test_fs.shape[2]))  
        X_test_scaled_not_reshaped = scaler.transform(X_test_reshaped)
        X_test_scaled = np.reshape(X_test_scaled_not_reshaped, X_test_fs.shape)

        # These are the schedule functions for the learning rate and 
        # momentum callbacks.

        lr_schedule_fn = get_one_cycle_lr_fn(
            epochs, 
            learning_rate,
            math.floor(epochs/10.0)
        )

        mom_schedule_fn = get_one_cycle_momentum_fn(
            epochs,
            learning_rate,
            math.floor(epochs/10.0)
        )

        # Setup code for the callbacks and optimizer. We use vanilla SGD
        # with momentum and 1cycle the learning rate.

        lr_scheduler = LearningRateScheduler(lr_schedule_fn, verbose=1)
        mom_scheduler = MomentumScheduler(mom_schedule_fn, verbose=1)

        optimizer = SGD(lr=lr_schedule_fn(0), momentum=mom_schedule_fn(0))
        callbacks = [lr_scheduler, mom_scheduler]


        # Get model object, based on hyperparams and input size.
        model = HARModel(
            batch_size = batch_size,
            OneD_kernel_size = 8,
            pooling="max",
            dense_size=1000,
            n_filters = 100,
            kernel_reg_param = 0.00005,
            input_shape=input_shape,
            classes=6,
            dpt=dropout
        )

        # Set up training process. We use categorical crossentropy
        # for the objective function, a standard choice for multilabel
        # classification.

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )        

        # Model training performed here. hist is a history
        # object that records the training performance by
        # epoch.

        hist = model.fit(
            x=X_train_scaled,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )

        # Write the training accuracy history by epoch to 
        # the log entry in case we need it later.

        log_entry += str(hist.history["acc"]) + "\n\n"

        # Generate test set predictions
        predicted_probabilities = model.predict(X_test_scaled)

        # Generate predicted class and ground truth vectors
        predictions = np.argmax(predicted_probabilities, axis=1)
        labels_flat = np.argmax(y_test, axis=1)

        # Calculate accuracy and add it to the log file.
        log_entry += "Accuracy: {0}\n\n".format(np.sum(predictions == labels_flat)/predictions.shape[0])

        # Classification report and confusion matrix allow for 
        # a better sense of model performance than just looking
        # at accuracy numbers.

        rpt = classification_report(labels_flat, predictions, labels=np.arange(len(labels)), target_names=labels)
        mat = confusion_matrix(labels_flat, predictions)

        # Write them to the log entry.

        log_entry += str(rpt) + "\n\n" + str(mat) + "\n\n"

    # Dump the results to the log. If the log already exists, we
    # append the new results to the file so we aren't losing
    # old runs.

    with open(experiment_log_dir + "UCI_HAR_Feature_Experiment.txt", 'a+') as outfile:
        outfile.write(log_entry)
