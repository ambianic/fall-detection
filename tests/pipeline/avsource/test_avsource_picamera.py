"""Test audio/video source pipeline element."""
import pytest
from ambianic.pipeline.avsource import picam
import logging
from io import BytesIO
import time
import os
from PIL import Image

log = logging.getLogger()
log.setLevel(logging.DEBUG)

class _TestPiCamera():

    def __init__(self, fail_read=False):
        _dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(
            _dir,
            '../ai/person.jpg'
        )
        self.img = Image.open(img_path, mode='r')
        self.fail_read = fail_read
        self.led = False

    def __iter__(self):
        log.debug("__iter__")
        return self

    def __next__(self):
        log.debug("__next__")
        pass

    def __enter__(self):
        log.debug("__enter__")
        return self

    def __exit__(self, type, value, tb):
        log.debug("__exit__")
        self.close()
        pass

    def start_preview(self):
        pass

    def capture_continuous(self, stream, format):
        self.capture(stream, format)
        return self

    def capture(self, stream: BytesIO, format: str):
        if self.fail_read:
            raise Exception("Read failed")
        self.img.save(stream, format=format)
        stream.seek(0)
        log.debug("acquired image stream")

    def close(self): 
        pass


# class _TestPiCameraFailure(_TestPiCamera):
#     def __init__(self):
#         super().__init__(fail_read=True)

class picamera_override():
    PiCamera = _TestPiCamera

# class picamera_override_failure():
#     PiCamera = _TestPiCameraFailure

def test_fail_import():
    picam.picamera_override = None
    with picam.Picamera() as cam:
        cam._get_camera()
        assert cam.has_failure()

def test_acquire():
    picam.picamera_override = picamera_override
    with picam.Picamera() as cam:
        assert not cam.has_failure()
        cam.start()
        time.sleep(1)
        raw = cam.acquire()
        assert raw is not None

    cam.stop()

# def test_acquire_failure():
#     picam.picamera_override = picamera_override_failure
#     cam = picam.Picamera()
#     assert not cam.has_failure()
#     failed = False
#     try:
#         cam.acquire()
#     except Exception:
#         failed = True
#     assert failed
#     cam.stop()
