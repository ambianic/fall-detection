#!/bin/bash

echo "Installing Ambianic dependencies"

# exit bash script on error
set -e

# verbose mode
set -x

# update apt-get and install sudo
apt-get update -y && apt-get install -y sudo

# check if python3 is installed
if python3 --version
then
  echo "python3 is already installed."
else
  # install python3 and pip3 which are not available by default on slim buster
  echo "python3 is not available from the parent image. Installing python3 now."
  sudo apt-get install -y python3 && apt-get install -y python3-pip
fi

# install numpy native lib
sudo apt-get install -y python3-numpy
sudo apt-get install -y libjpeg-dev zlib1g-dev


# install python dependencies
python3 -m pip install --upgrade pip
pip3 --version
pip3 install -r requirements.txt



echo "Installing tflite for x86 CPU"
if python3 --version | grep -q 3.8
then
  # pip3 install --force-reinstall https://github.com/google-coral/pycoral/releases/download/v1.0.1/tflite_runtime-2.5.0-cp38-cp38-linux_x86_64.whl
  pip3 install --force-reinstall https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp38-cp38-linux_x86_64.whl
else
  # pip3 install https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_x86_64.whl
  pip3 install --force-reinstall https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp37-cp37m-linux_x86_64.whl
fi

pip3 list
pip3 show tflite-runtime



# [Cleanup]
sudo apt-get -y autoremove

# remove apt-get cache
sudo  rm -rf /var/lib/apt/lists/*

# This is run automatically on Debian and Ubuntu, but just in case
sudo apt-get clean
