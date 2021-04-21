#!/bin/bash
set -ex

MY_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
MY_DIR=$(dirname "${MY_PATH}")

# test if coral usb stick is available
USB_DIR=/dev/bus/usb

if [ -d "$USB_DIR" ]; then
  USB_ARG="--device $USB_DIR"
else
  USB_ARG=""
fi

# check if there is an image update
docker pull ambianic/ambianic-edge:dev
# run dev image
docker run -it --rm \
  --name ambianic-edge-dev \
  --mount type=bind,source="$MY_DIR",target=/workspace \
  -p 8778:8778 \
  $USB_ARG \
  ambianic/ambianic-edge:dev /workspace/src/run-dev.sh
# on Mac, --net=host doesnt work as expected
#  --net=host \
