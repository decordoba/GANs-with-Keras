#!/usr/bin/env python3.5

import argparse
from time import clock
from datetime import timedelta
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers import Conv2D, MaxPooling2D
from keras import losses
from keras import optimizers
from keras_utils import get_params_from_shape
from keras_utils import flexible_neural_net
import numpy as np


def get_args():
    # Parser to allow fancy command arguments input
    parser = argparse.ArgumentParser(description="Test different models with lumps_matrix dataset")
    parser.add_argument('-f', '--folder', type=str, default=None,
                        help="Name of the folder where the results are saved. If not set, the "
                             "folder is named with the current date & time.")
    parser.add_argument('-ne', '--number_epochs', type=int, default=100,
                        help="Maximum number of epochs before termination. Default is 100.")
    parser.add_argument('-dr', '--data_reduction', type=int, default=1,
                        help="Number by which to divide the data used. For example, dr=3 means "
                        "only 1/3 of the data is used. Default is 1.")
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help="Size of every batch. Default is 32.")
    parser.add_argument("-s", "--silent", dest="silent", action="store_true", default=False,
                        help="Set verbose mode to false.")
    parser.add_argument("-dr", "--dry_run", dest="dry_run", action="store_true", default=False,
                        help="Run in dry run mode (does not fit the model).")
    return parser.parse_args()


if __name__ == "__main__":

    t = clock()

    args = get_args()

    (x_train, y_train), (x_test, y_test) = np.load("./datasets/lumps_matrix1/lumps_matrix1.npy")
    if args.data_reduction > 1:
        x_train = x_train[:x_train.shape[0] // args.data_reduction]
        y_train = y_train[:y_train.shape[0] // args.data_reduction]
        x_test = x_test[:x_test.shape[0] // args.data_reduction]
        y_test = y_test[:y_test.shape[0] // args.data_reduction]
    train_set = (x_train, y_train)
    test_set = (x_test, y_test)
    input_shape = train_set[0][1:]
    h, w, d = get_params_from_shape(input_shape)
    pixels_input = input_shape[0] * input_shape[1]
    optimizer = optimizers.Adam()
    loss = losses.mean_squared_error
    layers = [
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(units=pixels_input // 16, activation="relu"),
        Dense(units=pixels_input, activation="relu"),
        BatchNormalization(),
        Activation("relu"),
        Reshape((h // 4, w // 4,), input_shape=(pixels_input // 16,)),
        UpSampling2D(size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        UpSampling2D(size=(2, 2)),
        Conv2D(filters=1, kernel_size=(3, 3), activation="relu")
    ]

    flexible_neural_net(train_set, test_set, optimizer, loss, *layers, batch_size=args.batch_size,
                        epochs=args.number_epochs, early_stopping=10, location=args.folder,
                        verbose=not args.silent, dry_run=args.dry_run)

    print("\nTotal Time Taken: {} s".format(timedelta(seconds=clock() - t)))
