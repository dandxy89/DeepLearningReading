#!/bin/bash

# Pull Updates for Ubunutu
sudo apt-get update

# Install Updates for Ubunutu
sudo apt-get -y dist-upgrade

# Navigate to Keras
cd ~/keras

# Pull the Latest version of Keras
git pull

# Install the Latest Version
sudo python setup.py install

# Navigate to Theano
cd ~/Theano

# Pull the Latest version of Theano
git pull

# Install the Latest Version
sudo python setup.py install

