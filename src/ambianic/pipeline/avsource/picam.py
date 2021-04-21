import logging
from io import BytesIO, StringIO
import time
from PIL import Image
import threading
import queue

log = logging.getLogger(__name__)

picamera_override = None


class Picamera():

    def __init__(self, image_format='jpeg', queue_max_size=10):
        self.error = None
        self.format = image_format
        self.queue = queue.Queue(queue_max_size)
        self._stop = threading.Event()
        self.thread1 = threading.Thread(target=self.run, args=())

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, tb):
        self.stop()
        return self

    def start(self):
        self._stop.clear()
        self.thread1.start()

    def has_failure(self):
        return self.error is not None

    def _get_camera(self):
        if picamera_override is None:
            try:
                import picamera
                return picamera.PiCamera()
            except Exception as err:
                log.warning("Error loading picamera module: %s" % err)
                self.error = err
                return None
        else:
            return picamera_override.PiCamera()

    def run(self):
        cam = self._get_camera()
        if cam is None:
            # picam not available
            return
        with cam as camera:

            if self.has_failure():
                return None

            log.debug("Started Picamera")

            time.sleep(2)
            stream = BytesIO()
            for _ in camera.capture_continuous(stream, format=self.format):

                if self._stop.is_set():
                    log.debug("Stop requested")
                    break

                if not self.queue.full():
                    try:
                        self.queue.put(
                            Image.open(BytesIO(stream.getvalue())), 
                            block=False
                        )
                        log.debug("Queued capture")
                    except queue.Full:
                        pass
                    except Exception as ex:
                        log.error("Failed to add to queue: %s" % ex)

                stream.seek(0)
                stream.truncate()

            try:
                stream.close()
            except Exception as ex:
                log.error("Failed to close stream: %s" % ex)

    def acquire(self):
        try:
            # log.debug("queue len=%s" % self.queue.qsize())
            return self.queue.get(block=False)
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()
        self.queue = queue.Queue()
        self.thread1.join()
