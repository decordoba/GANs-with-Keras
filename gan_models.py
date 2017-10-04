from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras_utils import get_params_from_shape


def default_generator_model(input_size, output_shape):
    # input_size = size of random input for every batch
    # output_shape = shape of output image i.e. (28, 28, 1)
    # assumptions: output image is a multiple of 4. Reason: we upsample twice, so we multiply by 4

    h, w, d = get_params_from_shape(output_shape)
    if h %4 != 0 or w %4 != 0:
        raise ValueError("This generator can only return images whose side are multiples of 4")

    model = Sequential()
    model.add(Dense(units=1024, activation="tanh", input_shape=(input_size,)))
    model.add(Dense(units=128 * h * w // 16))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Reshape((h // 4, w // 4, 128), input_shape=(128 * h * w // 16,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="tanh", padding="same"))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=d, kernel_size=(5, 5), activation="tanh", padding="same"))
    return model  # Output: 1 image (d color channels)


def default_discriminator_model(input_shape):
    # input_shape is size of input image: (width, height, depth). i.e. mnist: (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation="tanh",
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=1024, activation="tanh"))
    model.add(Dense(units=1, activation="sigmoid"))
    return model  # Output: 1 value
