#!/bin/bash

# verbose mode
set -x

RELEASE_VERSION=$1
a=( ${RELEASE_VERSION//./ } )
MAJOR=${a[0]}
MINOR=${a[1]}
PATCH=${a[2]}
echo "RELEASE_VERSION=$RELEASE_VERSION"
echo "MAJOR=$MAJOR"
echo "MINOR=$MAJOR.$MINOR"
echo "PATCH=$PATCH"

# update version info in the ambianic python package setup.cfg
cp README.md src/
cd src
python3 setup.py setopt --command metadata --option version --set-value $RELEASE_VERSION

# verify if version is updated in setup.cfg
echo "verify if version is updated in setup.cfg"
cat setup.cfg
