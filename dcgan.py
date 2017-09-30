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


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def load_data(dataset, test_percentage=15):
    if dataset == "lumps":
        path = "./datasets/lumps/lumps.npy"
    else:
        raise KeyError("Unknown dataset: {}".format(dataset))
    data = np.load(path)
    num_samples = data.shape[0]
    num_test = num_samples * test_percentage // 100
    X_test = data[:num_test]
    y_test = np.ones(X_test.shape[0])
    X_train = data[num_test:]
    y_train = np.ones(X_train.shape[0])
    return (X_train, y_train), (X_test, y_test)


def train(BATCH_SIZE, dataset="mnist", epochs=100):
    # Load data from chosen dataset
    if dataset == "mnist" or dataset == "cifar10":
        if dataset == "mnist":
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
        else:
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
    elif dataset == "lumps":
        (X_train, y_train), (X_test, y_test) = load_data("lumps", test_percentage=0)
        min_val = X_train.min()
        vals_diff = (X_train.max() - min_val) / 2
        X_train = (X_train.astype(np.float32) + min_val - vals_diff) / vals_diff
    else:
        raise KeyError("Unknown dataset: {}".format(dataset))
    X_train = X_train[:, :, :, None]

    # Create folder where we will save all images
    now = datetime.now()
    date = "{}_{:02d}:{:02d}:{:02d}".format(now.date(), now.hour, now.minute, now.second)
    location = "training_dataset-{}_batch-size-{}_{}".format(dataset, BATCH_SIZE, date)
    try:
        os.makedirs(location)
    except OSError as e:
        pass  # The directory already exists

    # Create models G and D
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    # Create some variables to print nicely and save with best format
    batches = int(X_train.shape[0]/BATCH_SIZE)  # num batches every epoch
    len_batch = len(str(batches - 1))  # max num digits of a batch (used for printing)
    len_epoch = len(str(epochs - 1))  # max num digits of an epoch (used for printing)
    d_str = "Epoch: {{0:0{}d}}/{}.   Batch: {{:0{}d}}/{}.   D loss: {{:f}}".format(len_epoch, epochs, len_batch, batches)
    g_str = "Epoch: {{0:0{}d}}/{}.   Batch: {{:0{}d}}/{}.   G loss: {{:f}}".format(len_epoch, epochs, len_batch, batches)
    e_str = "{{:0{}d}}".format(len_epoch)

    # Discriminate and Generate iteratively for all epochs and batches
    d_losses = []
    g_losses = []
    for epoch in range(epochs):
        for batch in range(batches):
            # Generate fake images with generator
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            X_fake = g.predict(noise, verbose=0)
            # Get batch real training data
            X_real = X_train[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
            # Create dataset with labels: first half is real (1), second half is fake (0)
            X = np.concatenate((X_real, X_fake))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            # Calculate discriminator loss from dataset X and y
            d.trainable = True
            d_loss = d.train_on_batch(X, y)
            d_losses.append(d_loss)

            # Generate fake images with generator
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            # Calculate generator loss from noise
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            g_losses.append(g_loss)

            # Save an image and print some feedback messages
            if batch % 20 == 0:
                # Print feedback messages for G and D
                print(d_str.format(epoch, batch, d_loss))
                print(g_str.format(epoch, batch, g_loss))
                # Save sample image
                image = combine_images(X_fake)
                image = image * 127.5 + 127.5
                filename = "{}/{:03d}_{:03d}.png".format(location, epoch, batch)
                Image.fromarray(image.astype(np.uint8)).save(filename)

        # Save weights at the end of every epoch
        g.save_weights('generator.h5', True)
        d.save_weights('discriminator.h5', True)
        # Save D and G losses
        with open(location + "/result.yaml", "a") as f:
            f.write("epoch" + e_str.format(epoch) + ":\n")
            f.write("  g_loss: {}\n".format(g_loss))
            f.write("  d_loss: {}\n".format(d_loss))
            f.write("  g_losses: {}\n".format(g_losses))
            f.write("  d_losses: {}\n".format(d_losses))


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["train", "generate"], type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10", "lumps"], default="lumps", type= str)
    parser.add_argument("-n", "--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size, dataset=args.dataset)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
