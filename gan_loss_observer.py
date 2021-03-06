#!/usr/bin/env python3.5
from keras_plot import AGG  # Get it from github.com/decordoba/deep-learning-with-Keras
from matplotlib import pyplot as plt
from gan_utils import save_images_combined, load_dataset
from gan_utils import get_int_input, get_str_input
import argparse
import yaml
import shutil
import numpy as np


def get_config_manually():
    print("Enter the training configuration manually, or press ENTER to use the default values:")
    dct = dict()
    dct["dataset"] = get_str_input("Dataset", "lumps1")
    dct["batch_size"] = get_int_input("Batch Size", 128, 1)
    dct["batches"] = get_int_input("Batches every Epoch", 468, 1)
    return dct


def plot_dg(g_loss, d_loss, fig_num=0, filename=None, xaxis="Epoch", fig_clear=True,
            xaxis_multiplier=1, plot_mean=0):
    """
    Plots loss and accuracy in history
    If filename is None, the figure will be shown, otherwise it will be saved with name filename
    """
    # Plot epoch history for accuracy and loss
    if filename is None:
        plt.ion()
    fig = plt.figure(fig_num)
    if fig_clear:
        fig.clear()
    subfig = fig.add_subplot(111)
    if xaxis_multiplier == 1:
        subfig.plot(g_loss, label="Generator")
        subfig.plot(d_loss, label="Discriminator")
    else:
        y = range(0, len(g_loss) * xaxis_multiplier, xaxis_multiplier)
        subfig.plot(y, g_loss, label="Generator")
        y = range(0, len(d_loss) * xaxis_multiplier, xaxis_multiplier)
        subfig.plot(y, d_loss, label="Discriminator")
    subfig.set_title('Generator vs Discriminator Loss')
    subfig.set_xlabel(xaxis)
    subfig.set_ylabel('Loss')
    subfig.legend()
    if plot_mean < 0:  # for negative numbers, show only mean. Dirty, but fast
        plot_mean = -plot_mean
        fig.clear()
    if plot_mean > 0:
        g_mean = []
        d_mean = []
        num_samples = plot_mean
        for i, l in enumerate(g_loss):
            if i < num_samples:
                g_mean.append(np.mean(g_loss[:i + 1]))
            else:
                g_mean.append(np.mean(g_loss[i - num_samples + 1:i + 1]))
        for i, l in enumerate(g_loss):
            if i < num_samples:
                d_mean.append(np.mean(d_loss[:i + 1]))
            else:
                d_mean.append(np.mean(d_loss[i - num_samples + 1:i + 1]))
        if xaxis_multiplier == 1:
            subfig.plot(g_mean, label="Generator Mean")
            subfig.plot(d_mean, label="Discriminator Mean")
        else:
            y = range(0, len(g_mean) * xaxis_multiplier, xaxis_multiplier)
            subfig.plot(y, g_mean, label="Generator Mean")
            y = range(0, len(d_mean) * xaxis_multiplier, xaxis_multiplier)
            subfig.plot(y, d_loss, label="Discriminator Mean")
    if filename is None:
        plt.ioff()
    else:
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()


def plot_losses(folder=None, filename=None, save_img=False, image_summary=False,
                summary_content="both", manual_config=False):
    # Manage defaults
    if folder is None:
        folder = "."
    if folder[-1] != "/":
        folder += "/"
    if filename is None:
        filename = "result.yaml"
    d_losses = []
    g_losses = []
    d_loss = []
    g_loss = []
    # Open result.yaml
    with open(folder + filename) as f:
        try:
            print("Loading data from '{}'. This may take a while...".format(filename))
            result = yaml.load(f)
            keys = sorted(result.keys())
            n = 0
            for i in keys:
                d_loss.append(result[i]["d_loss"])
                g_loss.append(result[i]["g_loss"])
                d_losses += result[i]["d_losses"]
                g_losses += result[i]["g_losses"]
                n += 1
                if n % 1000 == 0:
                    print(n)
        except yaml.YAMLError as YamlError:
            print("There was an error parsing '{}'. Plotting aborted.".format(filename))
            print(YamlError)
            return
    # Open config.yaml or get config manually
    if manual_config:
        train_config = get_config_manually()
        dataset = train_config["dataset"]
        batch_size = train_config["batch_size"]
        num_batches = train_config["batches"]
    else:
        try:
            with open(folder + "config.yaml", "r") as f:
                try:
                    train_config = yaml.load(f)
                    dataset = train_config["dataset"]
                    batch_size = train_config["batch_size"]
                    num_batches = train_config["batches"]
                except yaml.YAMLError as YamlError:
                    print("There was a problem parsing 'config.yaml', no real images saved.")
                    print(YamlError)
                    dataset = None
                    num_batches = None
        except FileNotFoundError:
            print("File 'config.yaml' not found, no real images saved.")
            dataset = None
            num_batches = 468  # Hardcoded number, for backwards compatibility
    # Plot results
    if not AGG:
        plot_dg(g_loss=g_losses, d_loss=d_losses, xaxis="Batch")
        input("Press ENTER to continue")
        plot_dg(g_loss=g_losses, d_loss=d_losses, xaxis="Batch", plot_mean=50)
        input("Press ENTER to continue")
        plot_dg(g_loss=g_loss, d_loss=d_loss, xaxis="Batch", fig_clear=False,
                xaxis_multiplier=num_batches)
        input("Press ENTER to continue")
    # Save results images
    if save_img:
        output_filename = "GAN_loss.png"
        plot_dg(g_loss=g_losses, d_loss=d_losses, xaxis="Epoch", filename=folder + output_filename)
        print("Raw results saved in '{}'".format(folder + output_filename))
        output_filename = "GAN_loss_smooth.png"
        plot_dg(g_loss=g_losses, d_loss=d_losses, xaxis="Epoch", plot_mean=-50,
                filename=folder + output_filename)
        print("Averaged results saved in '{}'".format(folder + output_filename))

    # Save image summary. Will save all, only originals or only generated
    if image_summary:
        # Create folder where we will save all images
        folder_name = "image_summary"
        try:
            os.makedirs(folder + folder_name)
        except OSError:
            pass  # the folder already exists
        fake_images_saved = 0
        if summary_content == "both" or summary_content == "generated":
            all_imgs = sorted([f for f in os.listdir(folder) if f.endswith(".png") and
                               f[0].isdigit()])
            # Assume last image has the right suffix, like '_460.png'
            suffix = "_" + all_imgs[-1].split("_")[-1]
            summary_images = [f for f in all_imgs if f.endswith(suffix)]
            fake_images_saved = len(summary_images)
            for img_path in summary_images:
                shutil.copyfile(folder + img_path, folder + folder_name + "/" + img_path)
        real_images_saved = 0
        if summary_content == "both" or summary_content == "original":
            if dataset is not None:  # If config was loaded fine
                print("Loading dataset {}. This may take a while...".format(dataset))
                real_images = load_dataset(dataset, rng=(-1, 1))
                # Save 5 real images
                for i in range(5):  # Assume we will always have more than 5 batches
                    image_batch = real_images[i * batch_size:(i + 1) * batch_size]
                    save_images_combined(image_batch, folder + folder_name + "/" +
                                         "real_image_{}.png".format(i))
                    real_images_saved += 1

        print("{} images saved in {}".format(fake_images_saved + real_images_saved, folder +
                                             folder_name))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", default=None, type=str,
                        help="Location of 'result.yaml'. If unset, assumes current folder.")
    parser.add_argument("-r", "--result", default="result.yaml", type=str,
                        help="Filename with results. Default is 'result.yaml'.")
    parser.add_argument("-s", "--save", action="store_true", default=False,
                        help="Use this option to save image with loss graph to file.")
    parser.add_argument("-is", "--image_summary", action="store_true", default=False,
                        help="Use this to create folder with summary of generated images "
                             "every epoch.")
    parser.add_argument("-sc", "--summary_content", choices=["generated", "original", "both"],
                        type=str, default="both", help="Choose if image summary should contain "
                        "only real images, only generated imaged in every epoch, or both. "
                        "Default is 'both'.")
    parser.add_argument("-mc", "--manual_config", action="store_true", default=False,
                        help="If no config file exists, the required values can be entered "
                             "manually.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    plot_losses(args.folder, args.result, args.save, args.image_summary, args.summary_content,
                args.manual_config)
