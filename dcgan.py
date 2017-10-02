#!/usr/bin/env python3.5

"""
Modified from https://github.com/jacobgil/keras-dcgan
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist, cifar10
import numpy as np
from PIL import Image
from datetime import datetime
import argparse
import math
import os


def generator_model(input_size, output_shape):
    # input_size = size of random input for every batch, output_shape = shape of output image
    # assumptions: output image is a square image and the side is a multiple of 4
    # the reason for the multiple of 4 is because we do upsampling twice, so we multiply by 4
    # TODO: depth != 1???
    side = (output_shape[0], output_shape[1])
    if side[0] %4 != 0 or side[1] %4 != 0:
        raise ValueError("This generator can only return images whose side are multiples of 4")
    s4 = (side[0] // 4, side[1] // 4)
    model = Sequential()
    model.add(Dense(units=1024, activation="tanh", input_shape=(input_size,)))
    model.add(Dense(units=128 * s4[0] * s4[1]))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Reshape((s4[0], s4[1], 128), input_shape=(128 * s4[0] * s4[1],)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="tanh", padding="same"))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=1, kernel_size=(5, 5), activation="tanh", padding="same"))
    return model  # Output: 1 image


def discriminator_model(input_shape):
    # input_shape is size of input image: (width, height, depth). i.e. mnist: (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation="tanh",
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=1024, activation="tanh"))
    model.add(Dense(units=1, activation="sigmoid"))
    return model  # Output: 1 value


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]
    return image


def load_data(dataset):
    if dataset == "lumps":
        path = "./datasets/lumps/lumps.npy"
        X_train = np.load(path)  # We use all data for training
        min_val = X_train.min()
        max_val = X_train.max()
    elif dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        min_val = 0.0
        max_val = 255.0
    elif dataset == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        min_val = 0.0
        max_val = 255.0
    else:
        raise KeyError("Unknown dataset: {}".format(dataset))
    # Normalize data: All number will go from -1 to +1
    X_train = (X_train.astype(np.float32) - ((max_val + min_val) / 2.0)) / (max_val - min_val) * 2
    if len(X_train.shape) < 4:
        X_train = X_train[:, :, :, None]
    return X_train


def train(batch_size=128, dataset="mnist", epochs=100, noise_size=100, location=None):
    # Load data from chosen dataset
    X_train = load_data(dataset)
    print("Shape of '{}' dataset: {}".format(dataset, X_train.shape))

    # Plot real images from dataset
    try:
        from keras_plot import plot_images
        plot_images(X_train)
    except ImportError:
        pass

    # Create folder where we will save all images
    if location is None:
        now = datetime.now()
        date = "{}_{:02d}:{:02d}:{:02d}".format(now.date(), now.hour, now.minute, now.second)
        location = "training_dataset-{}_batch-size-{}_{}".format(dataset, batch_size, date)
    try:
        os.makedirs(location)
    except OSError as e:
        pass  # The directory already exists

    # Create models G and D
    input_shape = X_train.shape[1:]
    d = discriminator_model(input_shape)  # Maps image to label
    g = generator_model(noise_size, input_shape)  # Maps noise to image
    gan = generator_containing_discriminator(g, d)

    # Plot model used
    try:
        from keras_utils import plot_model
        plot_model(d, "discriminator_model.png", show_shapes=True, show_layer_names=False,
                   show_params=True)
        plot_model(g, "generator_model.png", show_shapes=True, show_layer_names=False,
                   show_params=True)
        plot_model(gan, "gan_model.png", show_shapes=True, show_layer_names=False,
                   show_params=True)
    except ImportError:
        pass
    # print summary of models
    g.summary()
    d.summary()
    gan.summary()


    # Compile models
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    gan.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    # Create some variables to print nicely and save with beautiful format
    batches = int(X_train.shape[0]/batch_size)  # num batches every epoch
    len_batch = len(str(batches))  # max num digits of a batch (used for printing)
    len_epoch = len(str(epochs))  # max num digits of an epoch (used for printing)
    d_str = "Epoch: {{:0{}d}}/{}.   Batch: {{:0{}d}}/{}.   D loss: {{}}".format(len_epoch,
                                                                    epochs, len_batch, batches)
    g_str = "Epoch: {{:0{}d}}/{}.   Batch: {{:0{}d}}/{}.   G loss: {{}}".format(len_epoch,
                                                                    epochs, len_batch, batches)
    e_str = "{{:0{}d}}".format(len_epoch)

    # Discriminate and Generate iteratively for all epochs and batches
    for epoch in range(1, epochs + 1):
        d_losses = []
        g_losses = []
        for batch in range(batches):
            # Generate fake images with generator
            noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
            X_fake = g.predict(noise, verbose=0)
            # Get batch real training data
            X_real = X_train[batch*batch_size:(batch+1)*batch_size]
            # Create dataset with labels: first half is real (1), second half is fake (0)
            X = np.concatenate((X_real, X_fake))
            y = [1] * batch_size + [0] * batch_size
            # Calculate discriminator loss from dataset X and y
            d.trainable = True
            d_loss = d.train_on_batch(X, y)
            d_losses.append(d_loss)

            # Generate fake images with generator
            noise = np.random.uniform(-1, 1, (batch_size, noise_size))
            # Calculate generator loss from noise
            d.trainable = False
            g_loss = gan.train_on_batch(noise, [1] * batch_size)
            g_losses.append(g_loss)

            # Save an image and print some feedback messages
            if (batch + 1) % 20 == 0:
                # Print feedback messages for G and D
                print(d_str.format(epoch, batch + 1, d_loss))
                print(g_str.format(epoch, batch + 1, g_loss))
                # Save sample image
                image = combine_images(X_fake)
                image = image * 127.5 + 127.5
                filename = "{}/{:03d}_{:03d}.png".format(location, epoch, batch + 1)
                Image.fromarray(image.astype(np.uint8)).save(filename)

        # Save weights at the end of every epoch
        g.save_weights(location + '/generator.h5', True)
        d.save_weights(location + '/discriminator.h5', True)
        # Save D and G losses
        with open(location + "/result.yaml", "a") as f:
            f.write("epoch" + e_str.format(epoch) + ":\n")
            f.write("  g_loss: {}\n".format(g_loss))
            f.write("  d_loss: {}\n".format(d_loss))
            f.write("  g_losses: {}\n".format(g_losses))
            f.write("  d_losses: {}\n".format(d_losses))


def generate(batch_size, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (batch_size*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, batch_size*20)
        index.resize((batch_size*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((batch_size,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(batch_size):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (batch_size, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["train", "generate"], type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10", "lumps"], default="lumps",
                        type=str)
    parser.add_argument("-f", "--folder", default=None, type=str)
    parser.add_argument("-n", "--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(batch_size=args.batch_size, dataset=args.dataset, location=args.folder)
    elif args.mode == "generate":
        generate(batch_size=args.batch_size, nice=args.nice)
