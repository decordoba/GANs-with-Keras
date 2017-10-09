import os
import numpy as np
from keras.utils import np_utils
from keras import backend
from keras.layers.wrappers import Wrapper
from keras.models import Sequential
from keras.datasets import mnist, cifar10
from keras_plot import plot_history
from datetime import datetime
from PIL import Image
import math
import pydot


GRAPHVIZ_NOT_INSTALLED = False


def plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True,
               show_params=False):
    """
    Extension of keras.utils.plot_model
    Plots model to an image, and also params of layers (original does not do it)
    """
    # To avoid program crashing, we will not try to plot the model without Graphviz
    global GRAPHVIZ_NOT_INSTALLED
    if GRAPHVIZ_NOT_INSTALLED:
        return False
    dot = model_to_dot(model, show_shapes, show_layer_names, show_params)
    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    try:
        dot.write(to_file, format=extension)
        return True
    except Exception:  # A generic Exception is raised if Graphviz is not installed
        GRAPHVIZ_NOT_INSTALLED = True
        print("Graphviz is not installed, therefore models will not be generated")
        return False

def model_to_dot(model, show_shapes=False, show_layer_names=True, show_params=False):
    """Converts a Keras model to dot format.

    # Arguments
        model: A Keras model instance.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        show_params: whether to display layer parameteres. (for now only works for some layers)

    # Returns
        A `pydot.Dot` instance representing the Keras model.
    """

    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    # Create graph nodes.
    for layer in layers:
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__
        if isinstance(layer, Wrapper):
            layer_name = '{}({})'.format(layer_name, layer.layer.name)
            child_class_name = layer.layer.__class__.__name__
            class_name = '{}({})'.format(class_name, child_class_name)

        # Create node's label.
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name)
        else:
            label = class_name

        if show_params:
            if "Conv2D" in class_name:
                label += "|filters: {}\nkernel_size: {}".format(layer.filters, layer.kernel_size)
                label += "\nactivation: {}\npadding: {}".format(str(layer.activation).split()[1],
                                                                layer.padding)
                label += "\nstrides: {}\nuse_bias: {}".format(layer.strides, layer.use_bias)
                try:
                    label += "\nkernel_reg: {}".format(str(layer.kernel_regularizer).split()[1])
                except IndexError:
                    label += "\nkernel_reg: {}".format(str(layer.kernel_regularizer))
                try:
                    label += "\nbias_reg: {}".format(str(layer.bias_regularizer).split()[1])
                except IndexError:
                    label += "\nbias_reg: {}".format(str(layer.bias_regularizer))
            elif "MaxPooling2D" in class_name or "AveragePooling2D" in class_name:
                label += "|pool_size: {}".format(layer.pool_size)
                label += "\nstrides: {}\npadding: {}".format(layer.strides, layer.padding)
            elif "Dropout" in class_name:
                label += "|rate: {}".format(layer.rate)
            elif "Dense" in class_name:
                label += "|units: {}\nactivation: {}".format(layer.units,
                                                              str(layer.activation).split()[1])
            elif "Activation" in class_name:
                label += "|activation: {}".format(str(layer.activation).split()[1])
            elif "BatchNormalization" in class_name:
                try:
                    label += "\ngamma_reg: {}".format(str(layer.gamma_regularizer).split()[1])
                except IndexError:
                    label += "\ngamma_reg: {}".format(str(layer.gamma_regularizer))
                try:
                    label += "\nbeta_reg: {}".format(str(layer.beta_regularizer).split()[1])
                except IndexError:
                    label += "\nbeta_reg: {}".format(str(layer.beta_regularizer))

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:
            try:
                outputlabels = str(layer.output_shape)
            except AttributeError:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels, outputlabels)

        node = pydot.Node(layer_id, label=label)
        dot.add_node(node)

    # Connect nodes with edges.
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot

def get_params_from_shape(shp):
    # Expects a 2 or 3 or 4 params shape like (64, 64), or (32, 32, 3), or (60000, 32, 32, 3)
    if len(shp) > 3:
        shp = shp[-3:]
    h = shp[0]
    w = shp[1]
    try:
        d = shp[2]
        if d > 4:
            # Assume max depth is 4, so we got shp like (60000, 28, 28)
            h = shp[1]
            w = shp[2]
            d = 1
    except IndexError:
        d = 1  # B/W
    return h, w, d

def load_dataset(dataset, rng=(-1, 1)): 
    # Returns data from dataset with shape (n, h, w, d), normalized from rng[0] to rng[1]
    min_val = None
    max_val = None
    if dataset == "lumps1" or dataset == "lumps":
        X_train = np.load("./datasets/lumps1/lumps1.npy")  # We use all data for training
    elif dataset == "lumps2":
        X_train = np.load("./datasets/lumps2/lumps2.npy")  # We use all data for training
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
    if min_val is None or max_val is None:
        min_val = X_train.min()
        max_val = X_train.max()
    # Normalize data: All number will go from rng[0] to rng[1]
    X_train = ((X_train.astype(np.float32) - min_val) / (max_val - min_val) * (rng[1] - rng[0])) + rng[0]
    if len(X_train.shape) < 4:
        X_train = X_train[:, :, :, None]  # Add depth to shape
    return X_train

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
            image[i*h:(i+1)*h, j*w:(j+1)*w, depth] = img[:, :]
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