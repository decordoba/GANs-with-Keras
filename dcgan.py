#!/usr/bin/env python3.5

"""
Modified from https://github.com/jacobgil/keras-dcgan
"""

from gan_models import *
from keras.models import Sequential
from keras.optimizers import SGD
from keras_utils import plot_model, get_params_from_shape  # Get it from github.com/decordoba/deep-learning-with-Keras
from gan_utils import save_images_combined, load_dataset
from gan_utils import get_current_time, get_int_input, get_str_input
from keras_plot import plot_images, AGG  # Get it from github.com/decordoba/deep-learning-with-Keras
import numpy as np
import yaml
from datetime import datetime
import argparse
import os


def get_config_manually(nice=True):
    print("Enter the training configuration manually, or press ENTER to use the default values:")
    dct = dict()
    dct["noise_size"] = get_int_input("Noise size", 100, 1)
    h = get_int_input("Input shape - height", 64, 1)
    w = get_int_input("Input shape - width", 64, 1)
    d = get_int_input("Input shape - depth", 1, 1, 5)
    dct["input_shape"] = (h, w, d)
    dct["generator_weights"] = [get_str_input("Weights file for Generator", "generator.h5")]
    dct["generator_model"] = get_str_input("Name Generator model", "default_generator_model")
    if nice:
        dct["discriminator_weights"] = [get_str_input("Weights file for Discriminator",
                                                      "discriminator.h5")]
        dct["discriminator_model"] = get_str_input("Name Discriminator model",
                                                   "default_discriminator_model")
    return dct


def train(dataset="mnist", batch_size=128, epochs=100, noise_size=100, location=None,
          generator_model=None, discriminator_model=None, g_optimizer=None,
          d_optimizer=None, gan_optimizer=None, save_model_frequency=10, save_images_frequency=100,
          d_repetitions=1, g_repetitions=1, freeze_d_in_gan=False):
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
        location = "{}_{:02d}.{:02d}.{:02d}".format(now.date(), now.hour, now.minute, now.second)
    if generator_model is None:
        generator_model = default_generator_model
    if discriminator_model is None:
        discriminator_model = default_discriminator_model

    # Force location to be in output/dataset/
    location = "output/{}/{}".format(dataset, location)
    # Create folder where we will save all images
    try:
        os.makedirs(location)
    except OSError:
        print("Error: the directory '{}' already exists.".format(location))
        return

    # Load data from chosen dataset
    x_train = load_dataset(dataset, rng=(-1, 1))
    print("Shape of '{}' dataset: {}".format(dataset, x_train.shape))

    # Plot real images from dataset (only if they can be shown (AGG == False))
    if not AGG:
        plot_images(x_train, invert_colors=True, title="Original data")

    # Create models G, D and GAN, and compile them
    input_shape = x_train.shape[1:]
    d = discriminator_model(input_shape)  # Maps image to label
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optimizer)
    g = generator_model(noise_size, input_shape)  # Maps noise to image
    g.compile(loss='binary_crossentropy', optimizer=g_optimizer)
    # For some reason, results are often better if d is not frozen
    if freeze_d_in_gan:
        d.trainable = False  # In gan model, discriminator weights are frozen
    gan = Sequential()
    gan.add(g)
    gan.add(d)
    gan.compile(loss='binary_crossentropy', optimizer=gan_optimizer)

    # Plot model used
    can_plot = True
    can_plot &= plot_model(d, location + "/discriminator_model.png", show_shapes=True,
                           show_layer_names=False, show_params=True)
    can_plot &= plot_model(g, location + "/generator_model.png", show_shapes=True,
                           show_layer_names=False, show_params=True)
    can_plot &= plot_model(gan, location + "/gan_model.png", show_shapes=True,
                           show_layer_names=False, show_params=True)
    if not can_plot:
        # Print summary of models if they could not be plotted and saved
        g.summary()
        d.summary()
        gan.summary()

    # Create some variables to print nicely and save with beautiful format
    batches = int(x_train.shape[0] / batch_size)  # num batches every epoch
    len_batch = len(str(batches))  # max num digits of a batch (used for printing)
    len_epoch = len(str(epochs))  # max num digits of an epoch (used for printing)
    d_str = "Epoch: {{:0{}d}}/{}.   Batch: {{:0{}d}}/{}.   D loss: {{}}".format(len_epoch,
                                                                                epochs, len_batch,
                                                                                batches)
    g_str = "Epoch: {{:0{}d}}/{}.   Batch: {{:0{}d}}/{}.   G loss: {{}}".format(len_epoch,
                                                                                epochs, len_batch,
                                                                                batches)
    e_str = "{{:0{}d}}".format(len_epoch)

    # Create file that saves information about model (read by generator)
    config_dict = {"input_shape": input_shape, "dataset": dataset, "batch_size": batch_size,
                   "epochs": epochs, "batches": batches, "noise_size": noise_size,
                   "generator_weights": [], "discriminator_weights": [],
                   "generator_model": generator_model.__name__,
                   "discriminator_model": discriminator_model.__name__}

    # Discriminate and Generate iteratively for all epochs and batches
    for epoch in range(1, epochs + 1):
        d_loss = None
        g_loss = None
        d_losses = []
        g_losses = []
        for batch in range(batches):
            for _ in range(d_repetitions):
                # Generate fake images with generator
                noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
                x_fake = g.predict(noise, verbose=0)
                # Get batch real training data
                x_real = x_train[batch * batch_size:(batch + 1) * batch_size]
                # Create dataset with labels: first half is real (1), second half is fake (0)
                x = np.concatenate((x_real, x_fake))
                y = [1] * batch_size + [0] * batch_size
                # Calculate discriminator loss from dataset X and y
                d_loss = d.train_on_batch(x, y)
                d_losses.append(d_loss)

            for _ in range(g_repetitions):
                # Generate fake images with generator
                noise = np.random.uniform(-1, 1, (batch_size, noise_size))
                # Calculate generator loss from noise
                g_loss = gan.train_on_batch(noise, [1] * batch_size)
                g_losses.append(g_loss)

            # Print feedback messages
            if (batch + 1) % 20 == 0:
                # Print feedback messages for G and D
                if (batch + 1) % 100 == 20:
                    print(get_current_time())
                print(d_str.format(epoch, batch + 1, d_loss))
                print(g_str.format(epoch, batch + 1, g_loss))
            # Save sample image
            if (batch + 1) % save_images_frequency == 0:
                save_images_combined(x_fake, "{}/{:03d}_{:03d}.png".format(location, epoch,
                                                                           batch + 1))

        # Save weights at the end of every epoch
        generator_filename = "generator{}.h5".format(((epoch - 1) // save_model_frequency) *
                                                     save_model_frequency)
        discriminator_filename = "discriminator{}.h5".format(((epoch - 1) // save_model_frequency)
                                                             * save_model_frequency)
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


def generate(batch_size=128, location=None, filename=None, nice=False, manual_config=False):
    # Set defaults
    if location is None:
        location = "."
    if filename is None:
        now = datetime.now()
        filename = "{}_{:02d}.{:02d}.{:02d}.png".format(now.date(), now.hour, now.minute,
                                                        now.second)

    # Should get everything else (generator_model, generator_weights_file, noise_size, input_shape,
    #                             discriminator_model, discriminator_weights_file) from config
    if not manual_config:
        with open(location + "/config.yaml", "r") as f:
            train_config = yaml.load(f)
    else:
        train_config = get_config_manually(nice)
    noise_size = train_config["noise_size"]
    input_shape = train_config["input_shape"]
    generator_weights_file = train_config["generator_weights"][-1]
    generator_model = globals()[train_config["generator_model"]]
    if nice:
        discriminator_weights_file = train_config["discriminator_weights"][-1]
        discriminator_model = globals()[train_config["discriminator_model"]]
    else:  # The only use for this is to mute annoying warnings from IDE
        discriminator_weights_file = None
        discriminator_model = None

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
        best_predictions = d_predictions.T[0].argsort()[-batch_size:][::-1]

        # Plot generated images (only if they can be shown (AGG == False))
        if not AGG:
            plot_images(generated_images, invert_colors=True, labels=d_predictions,
                        label_description="Prediction")

        # Get best images array and save to file
        h, w, d = get_params_from_shape(input_shape)
        best_images = np.zeros((batch_size, h, w, d), dtype=np.float32)
        for i, pred_idx in enumerate(best_predictions):
            best_images[i, :, :, 0] = generated_images[pred_idx, :, :, 0]
        save_images_combined(best_images, location + "/" + filename)
        print("Best generated images saved to {}".format(location + "/" + filename))
    else:
        noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
        print("Generating images...")
        generated_images = g.predict(noise, verbose=1)

        # Plot generated images (only if they can be shown (AGG == False))
        if not AGG:
            plot_images(generated_images, invert_colors=True)

        # Save to file
        save_images_combined(generated_images, location + "/" + filename)
        print("Generated images saved to {}".format(location + "/" + filename))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["train", "generate"], type=str, default="train",
                        help="'train' will train a GAN on the selected dataset. 'generate' will "
                             "generate new images from a trained GAN. Default is 'train'.")
    parser.add_argument("-f", "--folder", default=None, type=str,
                        help="Folder where to save data if training / extract data from if "
                             "generating.")
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10", "lumps1", "lumps2"],
                        default="lumps1", type=str, help="Only used in 'train' mode. "
                        "Dataset used for training. Default is 'lumps1'.")
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="Batch size while "
                        "training. Also number of samples saved in every image when training and "
                        "generating. Default is 128.")
    parser.add_argument("-ne", "--number_epochs", type=int, default=100, help="Only used in "
                        "'train' mode. Number of epochs the GAN will be trained. Default is 100.")
    parser.add_argument("-ns", "--noise_size", type=int, default=100, help="Only used in 'train' "
                        "mode. Length of random input vector used by the generator to generate "
                        "new images. Default is 100.")
    parser.add_argument("-sf", "--save_frequency", type=int, default=10, help="Only used in "
                        "'train' mode. How often a new weights file is created (how many epochs "
                        "between the different weight files are created). Default is 10.")
    parser.add_argument("-if", "--images_frequency", type=int, default=100, help="Only used in "
                        "'train' mode. How often new images are saved (how many batchs "
                        "between images are saved). Default is 100.")
    parser.add_argument("-rg", "--repetitions_generator", type=int, default=1, help="Only used "
                        "in 'train' mode. How many times we train the generator every batch. "
                        "Default is 1.")
    parser.add_argument("-rd", "--repetitions_discriminator", type=int, default=1, help="Only "
                        "used in 'train' mode. How many times we train the discriminator every "
                        "batch. Default is 1.")
    parser.add_argument("-fd", "--freeze_discriminator", dest="freeze", action="store_true",
                        default=False, help="Only used in 'train' mode. Freeze the discriminator"
                        " when training the full GAN. This should be the default mode, but I get"
                        " better results when it is not frozen.")
    parser.add_argument("-n", "--nice", dest="nice", action="store_true", default=False,
                        help="Only used in 'generate' mode. Generate more samples and show only "
                             "the ones that scored higher according to the discriminator.")
    parser.add_argument("-mc", "--manual_config", action="store_true", default=False,
                        help="Only used in 'generate' mode. If no config file exists, the required"
                             " values can be entered manually.")
    parser.add_argument("--agg", action="store_true", default=False,
                        help="Agg backend is used for matplotlib.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.agg:
        import matplotlib as mpl
        mpl.add("Agg")
        global AGG
        AGG = True
    if args.mode == "train":
        g_model = default_generator_model
        d_model = default_discriminator_model
        train(batch_size=args.batch_size, dataset=args.dataset, location=args.folder,
              epochs=args.number_epochs, noise_size=args.noise_size, freeze_d_in_gan=args.freeze,
              save_model_frequency=args.save_frequency, g_repetitions=args.repetitions_generator,
              d_repetitions=args.repetitions_discriminator, generator_model=g_model,
              save_images_frequency=args.images_frequency, discriminator_model=d_model)
    elif args.mode == "generate":
        generate(batch_size=args.batch_size, location=args.folder, filename=None, nice=args.nice,
                 manual_config=args.manual_config)
