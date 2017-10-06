#!/usr/bin/env python3.5

"""
Modified from https://github.com/jacobgil/keras-dcgan
"""

from gan_models import *
from keras.models import Sequential
from keras.optimizers import SGD
from keras_utils import save_images_combined, plot_model, load_dataset, get_current_time
from keras_plot import plot_images, AGG
import numpy as np
import yaml
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
        plot_images(X_train, invert_colors=True, title="Original data")

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

    # Create file that saves information about model (read by generator)
    config_dict = {"input_shape": input_shape, "dataset": dataset, "batch_size": batch_size,
                   "epochs": epochs, "noise_size": noise_size,
                   "generator_weights": [], "discriminator_weights": [],
                   "generator_model": generator_model.__name__,
                   "discriminator_model": discriminator_model.__name__}

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
                    print(get_current_time())
                print(d_str.format(epoch, batch + 1, d_loss))
                print(g_str.format(epoch, batch + 1, g_loss))
                # Save sample image
                save_images_combined(X_fake, "{}/{:03d}_{:03d}.png".format(location, epoch,
                                                                           batch + 1))

        # Save weights at the end of every epoch
        generator_filename = "generator{}.h5".format(((epoch - 1) // 10) * 10)
        discriminator_filename = "discriminator{}.h5".format(((epoch - 1) // 10) * 10)
        g.save_weights(location + "/" + generator_filename, True)
        d.save_weights(location + "/" + discriminator_filename, True)
        # Append weights locations to config if they have changed
        if (not len(config_dict["generator_weights"]) or
                    generator_filename != config_dict["generator_weights"][-1]):
            config_dict["generator_weights"].append(generator_filename)
        if (not len(config_dict["discriminator_weights"]) or
                    discriminator_filename != config_dict["discriminator_weights"][-1]):
            config_dict["discriminator_weights"].append(discriminator_filename)
        # Save config (overwrite every epoch)
        with open(location + "/config.yaml", "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        # Save D and G losses
        with open(location + "/result.yaml", "a") as f:
            f.write("epoch" + e_str.format(epoch) + ":\n")
            f.write("  g_loss: {}\n".format(g_loss))
            f.write("  d_loss: {}\n".format(d_loss))
            f.write("  g_losses: {}\n".format(g_losses))
            f.write("  d_losses: {}\n".format(d_losses))


def generate(batch_size=128, location=None, filename=None, nice=False):
    # Set defaults
    if location is None:
        location = "."
    if filename is None:
        now = datetime.now()
        filename = "{}_{:02d}.{:02d}.{:02d}.png".format(now.date(), now.hour, now.minute,
                                                        now.second)

    # Should get everything else (generator_model, generator_weights_file, noise_size, input_shape,
    #                             discriminator_model, discriminator_weights_file) from config
    with open(location + "/config.yaml", "r") as f:
        train_config = yaml.load(f)
    noise_size = train_config["noise_size"]
    input_shape = train_config["input_shape"]
    generator_weights_file = train_config["generator_weights"][-1]
    discriminator_weights_file = train_config["discriminator_weights"][-1]
    generator_model = globals()[train_config["generator_model"]]
    discriminator_model = globals()[train_config["discriminator_model"]]

    # Load and compile generator model
    g = generator_model(noise_size, input_shape)
    print("Loading generator weights...")
    g.load_weights(location + "/" + generator_weights_file)

    # Save images. Two modes: nice (generate many images and show only best) or default
    if nice:
        # Generate 20 times more samples than we want to show
        k = 20
        noise = np.random.uniform(-1, 1, size=(batch_size * k, noise_size))
        print("Generating images...")
        generated_images = g.predict(noise, verbose=1)

        # Create discriminator and only show the images that are more believable to D
        d = discriminator_model(input_shape)
        print("Loading discriminator weights...")
        d.load_weights(location + "/" + discriminator_weights_file)
        print("Labeling generated images...")
        d_predictions = d.predict(generated_images, verbose=1)
        # Indices of batch_size top predictions
        best_predictions = d_predictions.argsort()[-batch_size:][::-1]

        # Plot generated images (only if they can be shown (AGG == False))
        if not AGG:
            plot_images(generated_images, invert_colors=True, labels=d_predictions,
                        label_description="Prediction")

        # Get best images array and save to file
        h, w, d = get_params_from_shape(input_shape)
        best_images = np.zeros((batch_size, h, w, d), dtype=np.float32)
        for i, pred_idx in enumerate(best_predictions):
            best_images[i, :, :, 0] = generated_images[pred_idx, :, :, 0]
        save_images_combined(best_images, filename)
    else:
        noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
        print("Generating images...")
        generated_images = g.predict(noise, verbose=1)

        # Plot generated images (only if they can be shown (AGG == False))
        if not AGG:
            plot_images(generated_images, invert_colors=True)

        save_images_combined(generated_images, filename)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["train", "generate"], type=str, default="train")
    parser.add_argument("-bs", "--batch_size", type=int, default=128)
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10", "lumps1", "lumps2"],
                        default="lumps1", type=str)
    parser.add_argument("-f", "--folder", default=None, type=str)
    parser.add_argument("-n", "--nice", dest="nice", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(batch_size=args.batch_size, dataset=args.dataset, location=args.folder,
              generator_model=default_generator_model,
              discriminator_model=default_discriminator_model)
    elif args.mode == "generate":
        generate(batch_size=args.batch_size, location=args.folder, filename=None, nice=args.nice)
