#!/usr/bin/env bash

# Get required files from deep-learning-with-Keras (import as soft links)
git clone https://github.com/decordoba/deep-learning-with-Keras.git
ln -s deep-learning-with-Keras/keras_utils.py .
ln -s deep-learning-with-Keras/keras_plot.py .
ln -s deep-learning-with-Keras/keras_std.py .

# Install locally all requirements from pip (requires PIP installed, duh!)
pip3 install keras --user
pip3 install tensorflow --user
pip3 install matplotlib --user
pip3 install pydot --user
pip3 install h5py --user

echo "Dependencies installed"
