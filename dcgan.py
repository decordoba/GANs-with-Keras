#!/usr/bin/env python3.5

"""
Modified from https://github.com/jacobgil/keras-dcgan
"""

from gan_models import *
from keras.models import Sequential
from keras.optimizers import SGD
from keras_utils import combine_images, plot_model, load_dataset, getCurrentTime
from keras_plot import plot_images, AGG
import numpy as np
from PIL import Image
from datetime import datetime
import argparse
import os


def train(dataset="mnist", batch_size=128, epochs=100, noise_size=100, location=None,
          generator_model=None, discriminator_model=None, g_optimizer=None,
          d_optimizer=None, gan_optimizer=None):
    """
    default location: "output/dataset/data_time"
    default g_optimizer: SGD()
    default d_optimizer: SGD(lr=0.0005, momentum=0.9, nesterov=True)
    default gan_optimizer: SGD(lr=0.0005, momentum=0.9, nesterov=True)
    default generator_model: default_generator_model
    default discriminator_model: default_discriminator_model
    """
    # Set defaults
    if d_optimizer is None:
        d_optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    if g_optimizer is None:
        g_optimizer = SGD()
    if gan_optimizer is None:
        gan_optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    if location is None:
        now = datetime.now()
        date = "{}_{:02d}.{:02d}.{:02d}".format(now.date(), now.hour, now.minute, now.second)
        location = "output/{}/{}".format(dataset, date)
    if generator_model is None:
        generator_model = default_generator_model
    if discriminator_model is None:
        discriminator_model = default_discriminator_model

    # Create folder where we will save all images
    try:
        os.makedirs(location)
    except OSError as e:
        print("Error: the directory '{}' already exists.".format(location))
        return

    # Load data from chosen dataset
    X_train = load_dataset(dataset, rng=(-1, 1))
    print("Shape of '{}' dataset: {}".format(dataset, X_train.shape))


    # Plot real images from dataset (only if they can be shown (AGG == False))
    if not AGG:
        plot_images(X_train, invert_colors=True)

    # Create models G, D and GAN
    input_shape = X_train.shape[1:]
    d = discriminator_model(input_shape)  # Maps image to label
    g = generator_model(noise_size, input_shape)  # Maps noise to image
    gan = Sequential()
    gan.add(g)
    gan.add(d)

    # Compile models G, D and GAN
    g.compile(loss='binary_crossentropy', optimizer=g_optimizer)
    gan.compile(loss='binary_crossentropy', optimizer=gan_optimizer)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optimizer)

    # Plot model used
    can_plot = True
    can_plot &= plot_model(d, "discriminator_model.png", show_shapes=True, show_layer_names=False,
                           show_params=True)
    can_plot &= plot_model(g, "generator_model.png", show_shapes=True, show_layer_names=False,
                           show_params=True)
    can_plot &= plot_model(gan, "gan_model.png", show_shapes=True, show_layer_names=False,
                           show_params=True)
    if not can_plot:
        # Print summary of models if they could not be plotted and saved
        g.summary()
        d.summary()
        gan.summary()

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
                if (batch + 1) % 100 == 20:
                    print(getCurrentTime())
                print(d_str.format(epoch, batch + 1, d_loss))
                print(g_str.format(epoch, batch + 1, g_loss))
                # Save sample image
                image = combine_images(X_fake)
                image = image * 127.5 + 127.5
                filename = "{}/{:03d}_{:03d}.png".format(location, epoch, batch + 1)
                Image.fromarray(image.astype(np.uint8)).save(filename)

        # Save weights at the end of every epoch
        g.save_weights(location + '/generator{}.h5'.format(((epoch - 1) // 10) * 10), True)
        d.save_weights(location + '/discriminator{}.h5'.format(((epoch - 1) // 10) * 10), True)
        # Save D and G losses
        with open(location + "/result.yaml", "a") as f:
            f.write("epoch" + e_str.format(epoch) + ":\n")
            f.write("  g_loss: {}\n".format(g_loss))
            f.write("  d_loss: {}\n".format(d_loss))
            f.write("  g_losses: {}\n".format(g_losses))
            f.write("  d_losses: {}\n".format(d_losses))


def generate(batch_size, nice=False):
    pass
    # g = default_generator_model()
    # g.compile(loss='binary_crossentropy', optimizer="SGD")
    # g.load_weights('generator')
    # if nice:
    #     d = default_discriminator_model()
    #     d.compile(loss='binary_crossentropy', optimizer="SGD")
    #     d.load_weights('discriminator')
    #     noise = np.random.uniform(-1, 1, (batch_size*20, 100))
    #     generated_images = g.predict(noise, verbose=1)
    #     d_pret = d.predict(generated_images, verbose=1)
    #     index = np.arange(0, batch_size*20)
    #     index.resize((batch_size*20, 1))
    #     pre_with_index = list(np.append(d_pret, index, axis=1))
    #     pre_with_index.sort(key=lambda x: x[0], reverse=True)
    #     nice_images = np.zeros((batch_size,) + generated_images.shape[1:3], dtype=np.float32)
    #     nice_images = nice_images[:, :, :, None]
    #     for i in range(batch_size):
    #         idx = int(pre_with_index[i][1])
    #         nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
    #     image = combine_images(nice_images)
    # else:
    #     noise = np.random.uniform(-1, 1, (batch_size, 100))
    #     generated_images = g.predict(noise, verbose=1)
    #     image = combine_images(generated_images)
    # image = image*127.5+127.5
    # Image.fromarray(image.astype(np.uint8)).save(
    #     "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["train", "generate"], type=str, default="train")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10", "lumps1", "lumps2"],
                        default="lumps1", type=str)
    parser.add_argument("-f", "--folder", default=None, type=str)
    parser.add_argument("-n", "--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(batch_size=args.batch_size, dataset=args.dataset, location=args.folder,
              generator_model=default_generator_model,
              discriminator_model=default_discriminator_model)
    elif args.mode == "generate":
        generate(batch_size=args.batch_size, nice=args.nice)
