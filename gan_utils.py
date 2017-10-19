import os
import numpy as np
from keras.utils import np_utils
from keras import backend
from keras.datasets import mnist, cifar10
from keras_plot import plot_history  # Get it from github.com/decordoba/deep-learning-with-Keras
from datetime import datetime
from PIL import Image
import math
from keras_utils import plot_model, get_params_from_shape  # Get it from github.com/decordoba/deep-learning-with-Keras


def load_dataset(dataset, rng=(-1, 1)):
    # Returns data from dataset with shape (n, h, w, d), normalized from rng[0] to rng[1]
    min_val = None
    max_val = None
    if dataset == "lumps1" or dataset == "lumps":
        x_train = np.load("./datasets/lumps1/lumps1.npy")  # We use all data for training
    elif dataset == "lumps2":
        x_train = np.load("./datasets/lumps2/lumps2.npy")  # We use all data for training
    elif dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        min_val = 0.0
        max_val = 255.0
    elif dataset == "cifar10":
        (x_train, y_train), (X_test, y_test) = cifar10.load_data()
        min_val = 0.0
        max_val = 255.0
    else:
        raise KeyError("Unknown dataset: {}".format(dataset))
    if min_val is None or max_val is None:
        min_val = x_train.min()
        max_val = x_train.max()
    # Normalize data: All number will go from rng[0] to rng[1]
    x_train = ((x_train.astype(np.float32) - min_val) / (max_val - min_val) * (rng[1] - rng[0])) + rng[0]
    if len(x_train.shape) < 4:
        x_train = x_train[:, :, :, None]  # Add depth to shape
    return x_train


def combine_images(generated_images):
    # Generates image which is a combination of all the generated_images,
    # in a chessboard mode (in columns and rows)
    # Example: we get (64, 5, 5, 3), we return (40, 40, 3) = 8x8 images
    num = generated_images.shape[0]
    cols = int(math.sqrt(num))
    rows = int(math.ceil(float(num) / cols))
    h, w, d = get_params_from_shape(generated_images.shape[1:])
    image = np.zeros((h*rows, w*rows, d), dtype=generated_images.dtype)
    try:
        for depth in range(d):
            for index, img in enumerate(generated_images):
                i = int(index / cols)
                j = index % cols
                image[i*h:(i+1)*h, j*w:(j+1)*w, depth] = img[:, :, depth]
    except IndexError:
        # Assume only one color channel
        image = np.zeros((h*rows, w*rows, 1), dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index / cols)
            j = index % cols
            image[i*h:(i+1)*h, j*w:(j+1)*w, 0] = img[:, :]
    if d == 1:
        image = image[:, :, 0]
    return image  # Shape will always be (h, w, d) or (h, w) if d == 1


def save_images_combined(images, filename):
    """
    Save images as a single chessboard image with filename
    """
    # Assume range (-1, +1)
    image = combine_images(images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(filename)


def format_dataset(x_train, y_train, x_test=None, y_test=None, data_reduction=None,
                   to_categorical=False, ret_labels=False, verbose=False):
    """
    Reformat input: change dimensions, scale and cast so it can be fed into our model
    """
    # Check if x_test is passed
    test_available = True
    if x_test is None:
        x_test = np.array([])
        test_available = False
    if y_test is None:
        y_test = np.array([])
        test_available = False
    # Reduce number of examples (for real time debugging)
    if data_reduction is not None:
        x_train = x_train[:x_train.shape[0] // data_reduction]
        y_train = y_train[:y_train.shape[0] // data_reduction]
        x_test = x_test[:x_test.shape[0] // data_reduction]
        y_test = y_test[:y_test.shape[0] // data_reduction]

    # Get data parameters and save them as 'constants' (they will never change again)
    N_TRAIN = x_train.shape[0]
    N_TEST = x_test.shape[0]
    IMG_ROWS = x_train.shape[1]
    IMG_COLUMNS = x_train.shape[2]
    try:
        IMG_DEPTH = x_train.shape[3]
    except IndexError:
        IMG_DEPTH = 1  # B/W
    labels = np.unique(y_train)
    N_LABELS = len(labels)

    # Reshape input data
    if backend.image_data_format() == 'channels_first':
        X_train = x_train.reshape(N_TRAIN, IMG_DEPTH, IMG_ROWS, IMG_COLUMNS)
        X_test = x_test.reshape(N_TEST, IMG_DEPTH, IMG_ROWS, IMG_COLUMNS)
        input_shape = (IMG_DEPTH, IMG_ROWS, IMG_COLUMNS)
    else:
        X_train = x_train.reshape(N_TRAIN, IMG_ROWS, IMG_COLUMNS, IMG_DEPTH)
        X_test = x_test.reshape(N_TEST, IMG_ROWS, IMG_COLUMNS, IMG_DEPTH)
        input_shape = (IMG_ROWS, IMG_COLUMNS, IMG_DEPTH)

    # Convert data type to float32 and normalize data values to range [0, 1]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Reshape input labels
    if to_categorical:
        Y_train = np_utils.to_categorical(y_train, N_LABELS)
        Y_test = np_utils.to_categorical(y_test, N_LABELS)

    # Print information about input after reshaping
    if verbose:
        print("Training set shape:  {}".format(X_train.shape))
        print("Test set shape:      {}".format(X_test.shape))
        print("{} existing labels:  {}".format(N_LABELS, labels))

    if ret_labels:
        if not test_available:
            return (X_train, Y_train), input_shape, labels
        return (X_train, Y_train), (X_test, Y_test), input_shape, labels
    if not test_available:
        return (X_train, Y_train), input_shape
    return (X_train, Y_train), (X_test, Y_test), input_shape


def save_model_data(model=None, results=None, location=None, save_yaml=True, save_json=False,
                    save_image=True, save_weights=True, save_full_model=False, history=None):
    """
    Save in location (or current folder if location is None) information about the model.
    Results should be a dictionary with everything that has to be saved in result.yaml
    """
    if location is None:
        location = "."
    else:
        if not os.path.exists(location):
            os.makedirs(location)
    if model is not None:
        if save_image:
            plot_model(model, show_shapes=True, show_layer_names=False, show_params=True,
                       to_file=location + "/model.png")
        if save_yaml:
            with open(location + "/model.yaml", "w") as f:
                f.write(model.to_yaml())  # Load it with model = model_from_yaml(yaml_string)
        if save_json:
            with open(location + "/model.json", "w") as f:
                f.write(model.to_json())  # Load it with model = model_from_json(json_string)
        if save_weights:
            model.save_weights(location + '/weights.h5')  # Load it with model.load_weights('w.h5')
        if save_full_model:
            model.save(location + '/model.h5')  # Load it with model = load_model('my_model.h5')
    if history is not None:
        plot_history(history, filename=location + "/history.png")
    if results is not None:
        result = ""
        for k in results:
            result += "{}: {}\n".format(k, results[k])
        with open(location + "/result.yaml", "w") as f:
            f.write(result)


def get_current_time(time=True, date=False):
    # Get time (and date) in a human-readable format (yyyy-mm-dd hh:mm:ss)
    now = datetime.now()
    s = ""
    if date:
        s += "{} ".format(now.date())
    if time:
        s += "{:02d}:{:02d}:{:02d}".format(now.hour, now.minute, now.second)
    return s.strip()


def get_int_input(text, default=None, min_val=None, max_val=None):
    # Get number entered by user. Continue asking until number is valid and between min_val and
    # max_val. Also, if ENTER is selected, the default number is returned.
    while True:
        try:
            n = input(text + " [{}]:  ".format(default))
            if len(n) == 0:
                return default
            n = int(n)
            if (min_val is None or n >= min_val) and (max_val is None or n < max_val):
                return n
        except ValueError:
            pass


def get_str_input(text, default=None):
    # Get string entered by user. If ENTER is selected, the default string is returned.
    s = input(text + " [{}]:  ".format(default))
    if len(s) == 0:
        return default
    return s
