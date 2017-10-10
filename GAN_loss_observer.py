#!/usr/bin/env python3.5
import matplotlib as mpl
import os
AGG = False
r1 = os.system('python3 -c "import matplotlib.pyplot as plt;plt.figure()"')  # Linux
r2 = os.system('py -c "import matplotlib.pyplot as plt;plt.figure()"')  # Windows
# This line allows mpl to run with no DISPLAY defined
if r1 != 0 and r2 != 0:
    print("$DISPLAY not detected, matplotlib set to use 'Agg' backend")
    mpl.use('Agg')
    AGG = True
from matplotlib import pyplot as plt
from keras_utils import save_images_combined, load_dataset
from keras_utils import get_int_input, get_str_input
import argparse
import yaml
import shutil
import numpy as np


def get_config_manually():
    print("Enter the training configuration manually, or press ENTER to use the default values:")
    dct = dict()
    dct["dataset"] = get_str_input("Dataset", "lumps1")
    dct["batch_size"] = get_int_input("Batch Size", 128, 1)
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
                manual_config=False):
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
    with open(folder + filename) as f:
        try:
            print("Loading data from {}. This may take a while...".format(filename))
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
            print("Data from {} loaded".format(filename))
        except yaml.YAMLError as YamlError:
            print("There was an error parsing '{}'. Plotting aborted.".format(filename))
            print(YamlError)
            return
    if not AGG:
        plot_dg(g_loss=g_losses, d_loss=d_losses, xaxis="Batch")
        input("Press ENTER to continue")
        plot_dg(g_loss=g_losses, d_loss=d_losses, xaxis="Batch", plot_mean=50)
        input("Press ENTER to continue")
        plot_dg(g_loss=g_loss, d_loss=d_loss, xaxis="Batch", fig_clear=False, xaxis_multiplier=468)
        input("Press ENTER to continue")
    if save_img:
        output_filename = "GAN_loss.png"
        plot_dg(g_loss=g_losses, d_loss=d_losses, xaxis="Batch", filename=folder + output_filename)
        print("Results saved in {}".format(folder + output_filename))
        output_filename = "GAN_loss_smooth.png"
        plot_dg(g_loss=g_losses, d_loss=d_losses, xaxis="Batch", plot_mean=50,
                filename=folder + output_filename)
        print("Averaged results saved in {}".format(folder + output_filename))

    if image_summary:
        # Create folder where we will save all images
        folder_name = "image_summary"
        try:
            os.makedirs(folder + folder_name)
        except OSError:
            pass  # the folder already exists
        all_imgs = sorted([f for f in os.listdir(folder) if f.endswith(".png") and f[0].isdigit()])
        # Assume last image has the right suffix, like '_460.png'
        suffix = "_" + all_imgs[-1].split("_")[-1]
        summary_images = [f for f in all_imgs if f.endswith(suffix)]
        for img_path in summary_images:
            shutil.copyfile(folder + img_path, folder + folder_name + "/" + img_path)
        real_images_saved = 0
        if manual_config:
            train_config = get_config_manually()
            dataset = train_config["dataset"]
            batch_size = train_config["batch_size"]
        else:
            try:
                with open(folder + "config.yaml", "r") as f:
                    try:
                        train_config = yaml.load(f)
                        dataset = train_config["dataset"]
                        batch_size = train_config["batch_size"]
                    except yaml.YAMLError as YamlError:
                        print("There was a problem parsing 'config.yaml', no real images saved.")
                        print(YamlError)
                        dataset = None
            except FileNotFoundError:
                print("File 'config.yaml' not found, no real images saved.")
                dataset = None
        if dataset is not None:
            print("Loading dataset {}. This may take a while...".format(dataset))
            real_images = load_dataset(dataset, rng=(-1, 1))
            # Save 5 real images
            for i in range(5):  # Assume we will always have more than 5 batches
                image_batch = real_images[i * batch_size:(i + 1) * batch_size]
                save_images_combined(image_batch, folder + folder_name + "/" +
                                     "real_image_{}.png".format(i))
                real_images_saved += 1

        print("{} images saved in {}".format(len(summary_images) + real_images_saved, folder +
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
    parser.add_argument("-mc", "--manual_config", action="store_true", default=False,
                        help="If no config file exists, the required are entered manually.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    plot_losses(args.folder, args.result, args.save, args.image_summary, args.manual_config)
