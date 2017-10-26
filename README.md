# GANs with Keras

Implementation of Generative Adversarial Networks (GANs) using Keras 2. Generate original artificial images, like the ones that you would find in another dataset of real images, which are completely new.

Two adversarial deep neural networks (the Generator and the Discriminator) are trained on real images, in order to produce artificial images that look real.

The Generator model tries to produce images that look real and are thought to be real by the Discriminator. The Discriminator model tries to tell real images apart from the artificial images generated by the Generator. Therefore, both models become better epoch after epoch at their job, as the other model improves; this is a game of two adversaries, G and D, and for this reason we call them generative adversarial networks. In theory, if we choose good models and train both networks long enough, the Generator will end up generating images so good that they will be impossible to differenciate from real images.

This project's code is based on this repository: https://github.com/jacobgil/keras-dcgan

## Sample results

I will add some images here soon.

## Installation

Run the following to install all the dependencies:

`bash get_dependencies.sh`

If your system cannot run bash commands, simply clone the `deep-learning-with-Keras` repository, copy at the same level of `dcgan.py` all the files starting with `keras_` and install the required pip wheels:

```
git clone https://github.com/decordoba/deep-learning-with-Keras.git
cp deep-learning-with-Keras/keras_* .
pip3 install keras tensorflow matplotlib pydot h5py --user
```

Make sure that once this is done, `python3 dcgan.py -d mnist` runs successfully. Refer to the Troubleshooting section if you run into any issue (you probably need to install a few more things).

## Usage
### `dcgan.py`
`dcgan.py` will train the GAN or generate new images from an already trained GAN.

Run `python3 dcgan.py -h` to see how to use the file:
```
$ python3 dcgan.py --help
usage: dcgan.py [-h] [-m {train,generate}] [-f FOLDER]
                [-d {mnist,cifar10,lumps1,lumps2}] [-bs BATCH_SIZE]
                [-ne NUMBER_EPOCHS] [-ns NOISE_SIZE] [-sf SAVE_FREQUENCY] [-n]
                [-mc]

optional arguments:
  -h, --help            show this help message and exit
  -m {train,generate}, --mode {train,generate}
                        'train' will train a GAN on the selected dataset.
                        'generate' will generate new images from a trained
                        GAN. Default is 'train'.
  -f FOLDER, --folder FOLDER
                        Folder where to save data if training / extract data
                        from if generating.
  -d {mnist,cifar10,lumps1,lumps2}, --dataset {mnist,cifar10,lumps1,lumps2}
                        Only used in 'train' mode. Dataset used for training.
                        Default is 'lumps1'.
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size while training. Also number of samples
                        saved in every image when training and generating.
                        Default is 128.
  -ne NUMBER_EPOCHS, --number_epochs NUMBER_EPOCHS
                        Only used in 'train' mode. Number of epochs the GAN
                        will be trained. Default is 100.
  -ns NOISE_SIZE, --noise_size NOISE_SIZE
                        Only used in 'train' mode. Length of random input
                        vector used by the generator to generate new images.
                        Default is 100.
  -sf SAVE_FREQUENCY, --save_frequency SAVE_FREQUENCY
                        Only used in 'train' mode. How often a new weights
                        file is created (how many epochs between the different
                        weight files are created). Default is 10.
  -n, --nice            Only used in 'generate' mode. Generate more samples
                        and show only the ones that scored higher according to
                        the discriminator.
  -mc, --manual_config  Only used in 'generate' mode. If no config file
                        exists, the required values can be entered manually.
```

There are two modes: **train** and **generate**. 

In **train mode**, the dataset, the batch size, the number of epochs, the noise size, the folder where the result will be saved, and the weights checkpoint frequency can be set from command line. 

Just try it out! Run:

`python3 dcgan.py -d mnist`

or, if you want to be more specific, run:

`python3 dcgan.py -m trawn -d mnist -f my_first_gan -bs 64 -ne 100 -ns 100 -sf 10`

This will trigger several actions:
1. A folder will be created in `output/mnist/` called `YYYY-MM-DD_hh.mm.ss` (the current date), unless you select a different name using the `-f` option.
2. The selected dataset (mnist in the above case) will be loaded. This may take a few minutes.
3. A window will appear showing all images in the dataset, one at a time. We can see each of them typing ENTER successively, or stop the samples viewer typing q + ENTER.
4. The generator and discriminator models are created and compiled. The models used can be selected modifying the arguments passed in the main function. The default models are found in `gan_models.py`.
5. Images of the generator, discriminator and GAN models are saved in the `output/dataset/` with the names `generator-model.png`, `discriminator_model.png` and `gan_model.png` (`pydot` and `graphviz` are required for this. If they are not found, this step will be skipped and a summary of the models will be printed in console).
6. Enter the main loop where the GAN will be trained. Every 20 batches, some feedback is printed (the epoch number, the batch number, the loss, the time, etc.). At the end of every epoch, the generator and discriminator weights are saved (so we can generate and discriminate images using those models), the results are saved in a `results.yaml` file, and the configuration is updated in the `config.yaml` file. `config.yaml` will be used in the generate mode to know the setup used originally to train the GAN. `results.yaml` will be used by `gan_loss_observer.py` to plot a graph of the loss evolveution.
7. Once we have looped over all the epochs, the execution ends. This process can take several hours, to speed it up reduce the number of epochs, or simplify the model used.

**Notes:**
* The generator creates an image from a noise vector of length `noise_size`. We can think of this as the generator mapping a low dimensional input to a high dimensional image.
* The `save_frequency` option tells the system how often to save a weights file. By default, at the end of every epoch a weights file is created, which will overwrite the previous weights file. The point of `save_frequency` is to also save copies of weights during the training process. Therefore, if we have 100 epochs and `save_frequency` is 10, at the end of training we will have 10 weights files: one for epoch 10, another for epoch 20, 30, 40, ..., 100. The point of this is that sometimes the network will start performing worse after some epoch and if we store some weights checkpoint, we can go back to the model around that point. Set `sf` to 1 if you want a checkpoint for every epoch, and set it to `ne` to keep only the last weights file.

**Generate mode** only works if we have previously trained a GAN and use the option `-f` to select where the weights and configuration files were saved. This option will generate new images using the selected models. 

Run it:

`python3 dcgan.py -m generate -f output/mnist/my_first_gan`

or, if you want to generate more samples and plot only the ones that the discriminator prefers (`--nice` option): 

`python3 dcgan.py -m generate -f my_first_gan -bs 64 --nice`

This will trigger several actions:
1. If the `--manual_config` option is selected, a dialog will appear to enter the configuration used during training manually. Else, all the information will be loaded from `config.yaml`. The `-mc` option is used to support older versions of the code where the config file was not created.
2. The generator and discriminator models are created, according to what is said in the `config.yaml` file, and the weights are loaded to them, from the files indicated in the `config.yaml`.
3. If the `--nice` modifier is applied, `batch_size * 20` images are created with the generator, and the best `batch_size` images (those best classified by the discriminator) are shown in a pop-up window, and saved. If the `--nice` modifier is not applied, only `batch_size` images are generated, and they are all shown and saved. The images file will be named with the current date and time.

**Notes:**
* The `config.yaml` file contains crucial information to build the generator model: the noise vector length, the shape of the output image, the location of the weights files and the model used as the generator. This file can be modified to load a different weights file, or change the dimensions of the output image, but do with care as the generator relies on this information to generate the images. Always backup your files before modifying them, or use th `-mc` option to insert the configuration manually instead of changing the `config.yaml`file.

### `gan_loss_observer.py`
Observe losses and how they change for both the generator and discriminator after training a GAN.

Run `python3 gan_loss_observer.py -h` to see how to use the file:
```
$ python3 gan_loss_observer.py --help
usage: gan_loss_observer.py [-h] [-f FOLDER] [-r RESULT] [-s] [-is]
                            [-sc {generated,original,both}] [-mc]

optional arguments:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        Location of 'result.yaml'. If unset, assumes current
                        folder.
  -r RESULT, --result RESULT
                        Filename with results. Default is 'result.yaml'.
  -s, --save            Use this option to save image with loss graph to file.
  -is, --image_summary  Use this to create folder with summary of generated
                        images every epoch.
  -sc {generated,original,both}, --summary_content {generated,original,both}
                        Choose if image summary should contain only real
                        images, only generated imaged in every epoch, or both.
                        Default is 'both'.
  -mc, --manual_config  If no config file exists, the required values can be
                        entered manually.
```

Run it:

`python3 gan_loss_observer.py -f output/mnist/my_first_gan --save --image_summary`

This will show a graph in a new window with the loss curves of the generator and discriminator in the selected folder, save the graph in the same folder, and create a `image_summary` folder where the generated images at the end of every epoch are saved, as well as some real images for comparison.

### `gan_models.py`

Models to be used should be saved here. Then, they can be chosen from the main function in `dcgan.py`, substituting the default models by the new models. Right now the `dcgan.py` main() looks like:

```python
if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        g_model = default_generator_model
        d_model = default_discriminator_model
        train(... arguments here ...)
    elif args.mode == "generate":
        generate(... arguments here ...)
```

If you create new models in `gan_models.py` called `my_d_model` and `my_g_model`, simply change the `g_model` and `d_model` lines to be:
```python
    if args.mode == "train":
        g_model = my_g_model
        d_model = my_d_model
        train(... arguments here ...)
```

The model names will be saved in the `config.yaml` file, so there is no need to pass the model to the generate mode. Beware that if you change the model once trained, you may break the generate mode for the created file, as it needs the original models to generate new images.

# Troubleshooting

#### I get an error that says "No module named _tkinter", and no images are shown when I run `dcgan.py -d mnist`.

Your python probably does not come with tkinter. Make sure you are using python 3.4 or above, and download the right python:

**Ubuntu:**

`sudo apt install python3-tk`

**CentOS:**

```
sudo yum install tkinter tk-devel openssl-devel
# Make and install python again. In the case of 3.5:
wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
tar xzf Python-3.5.2.tgz
cd Python-3.5.2
./configure
make altinstall
```

A Google search will tell you more about the problem and how to solve it in other machines.


#### I see a message that says "Graphviz is not installed, therefore models will not be generated".

You need to install graphviz:

Ubuntu:

`sudo apt install graphviz`

CentOS:

`sudo yum install graphviz` 

Once more, the solution may be different for every operating system. This error happens because pydot depends on graphviz. Anyway, the program will continue running anyway in case you don't have admin privileges to install anything, but then you will not get a nice image of the model.

#### I get an error message like "FileNotFoundError: [Errno 2] No such file or directory: './datasets/lumps1/lumps1.npy'" or similar. 

You are trying to access a dataset that you don't have. The datasets `lumps1` and `lumps2` are datasets that I generated for my research and are too big for Github. I may upload them in the future, but I doubt it. You can generate new datasets similar to mine using the tools in the `dataset_generator` folder. Add your new datasets to the `load_dataset` function in gan_utils. Also, the cifar10 dataset has not been fully tested. Be warned!

#### No images are ever displayed, and I see the message "$DISPLAY not detected, matplotlib set to use 'Agg' backend".

The "Agg" backend is being used in `matplotlib`. This is used to allow `matplotlib` to work on systems without a $DISPLAY, like when we SSH into another system without the `-X` or `-Y` option, if inside screen, or if using Putty in Windows without X11 forwarding and Xming. This may also happen if tkinter is not working, so make sure _tkinter works before worrying about your $DISPLAY. To solve this problem, make sure your display is detected: a new window should be created when you run `python3 -c "import matplotlib.pyplot as plt;plt.figure();plt.show()"`.

#### I see a message like "python3 is not recognized as an internal or external command" or "sh: py: command not found".

This is not an error, everything works fine. What I do to know if I should activate "Agg" is to run:

```python
r1 = os.system('python3 -c "import matplotlib.pyplot as plt;plt.figure()"')  # Linux
r2 = os.system('py -c "import matplotlib.pyplot as plt;plt.figure()"')  # Windows
# This if allows mpl to run with no DISPLAY defined
if r1 != 0 and r2 != 0:  # If both lines fail, assume no DISPLAY
    print("$DISPLAY not detected, matplotlib set to use 'Agg' backend")
    mpl.use('Agg')
    AGG = True
```

Because I don't know if you are in Windows or Linux, I run the above `py` and `python3` commands, and see if any of them succeeds. If they do, I know I can use `matplotlib` without "Agg", else I have to activate the Agg backend, which will not show ever any visual window that requires $DISPLAY. Unfortunately I have not found any way of hiding the error messages, so they will always be shown at the beginning of your execution. Sorry!
