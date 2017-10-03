#!/usr/bin/env python3.5
from matplotlib import pyplot as plt
import argparse
import yaml
import os
import numpy as np


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


def plot_losses(folder, filename, save_img=False):
    if folder is not None:
        os.chdir(folder)
    if filename is None:
        filename = "result.yaml"
    d_losses = []
    g_losses = []
    d_loss = []
    g_loss = []
    with open(filename) as f:
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
            print("There was an error parsing 'result.yaml'. Plotting aborted.")
            print(YamlError)
            if folder is not None:
                os.chdir("./..")
            return
    plot_dg(g_loss=g_losses, d_loss=d_losses, xaxis="Batch")
    input("Press ENTER to continue")
    plot_dg(g_loss=g_losses, d_loss=d_losses, xaxis="Batch", plot_mean=50)
    input("Press ENTER to continue")
    plot_dg(g_loss=g_loss, d_loss=d_loss, xaxis="Batch", fig_clear=False, xaxis_multiplier=468)
    input("Press ENTER to continue")
    if save_img:
        plot_dg(g_loss=g_losses, d_loss=d_losses, xaxis="Batch", filename="GAN_loss.png")

    if folder is not None:
        os.chdir("./..")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", default=None, type=str)
    parser.add_argument("-r", "--result", default=None, type=str)
    parser.add_argument("-s", "--save", default=False, type=bool)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    plot_losses(args.folder, args.result, args.save)