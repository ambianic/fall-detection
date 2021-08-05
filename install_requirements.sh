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

pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime
pip3 list

# [Cleanup]
sudo apt-get -y autoremove

# remove apt-get cache
sudo  rm -rf /var/lib/apt/lists/*

# This is run automatically on Debian and Ubuntu, but just in case
sudo apt-get clean
