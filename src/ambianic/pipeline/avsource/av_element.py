"""Audio Video sourcing to an Ambianic pipeline."""

import logging
import time
import threading
import multiprocessing
import queue
from PIL import Image
from io import BytesIO
import requests
from ambianic.util import stacktrace
from ambianic.pipeline import PipeElement
from ambianic.pipeline.avsource import gst_process
from ambianic.pipeline.avsource.picam import Picamera

log = logging.getLogger(__name__)


MIN_HEALING_INTERVAL = 5


class AVSourceElement(PipeElement):
    """
    Pipe element that handles a wide range of media input sources.

    Detects media input source, processes and passes on normalized raw
    image samples to the next pipe element.
    """

    def __init__(self, uri=None, type=None, live=False, **kwargs):
        """Create an av source element with given configuration.

        :Parameters:
        ----------
        source_conf : dict
            uri: string (examples
                uri: rtsp://somehost/ipcam/channel0
                uri: http://somehost/ipcam/sample.jpg
                )
            type: string (video, audio or image)
            live: boolean (True if the source is a live stream)
                When live is True AVSourceElement source element will
                keep trying to reconnect
                in case there is disruption of the source stream
                until explicit stop() is requested of the element.
        """
        super().__init__(**kwargs)

        assert uri

        element_conf = dict(kwargs)
        element_conf['uri'] = uri
        element_conf['type'] = type
        element_conf['live'] = live

        # pipeline source info
        self._source_conf = element_conf
        self._is_live = live
        self._gst_process = None
        self._gst_out_queue = None
        self._gst_process_stop_signal = None
        self._gst_process_eos_reached = None
        # protects access to gstreamer resources in rare cases
        # such as supervised healing requests
        self._healing_in_progress = threading.RLock()
        # ensure healing requests are reasonably spaced out
        self._latest_healing = time.monotonic()

    def _on_new_sample(self, sample=None):
        log.debug('Input stream received new gst sample.')
        assert sample
        sample_type = sample['type']
        # only image type supported at this time
        assert sample_type == 'image'
        # make sure the sample is in RGB format
        sample_format = sample['format']
        assert sample_format == 'RGB'
        width = sample['width']
        height = sample['height']
        sample_bytes = sample['bytes']
        img = Image.frombytes(sample_format, (width, height),
                              sample_bytes, 'raw')
        # pass image sample to next pipe element, e.g. ai inference
        log.debug('Input stream sending sample to next element.')
        self.receive_next_sample(image=img)

    def _get_gst_service_starter(self):
        return gst_process.start_gst_service

    def _get_sample_queue(self):
        q = multiprocessing.Queue(3)
        return q

    def fetch_img(self, session=None, url=None) -> Image:
        assert url
        r = requests.get(url)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))
        return img

    def _on_fetch_img_exception(self, _exception=None):
        pass

    def _fetch_img_exception_recovery(self):
        log.debug("Pausing for a moment to let remote network issues settle")
        time.sleep(1)

    def _run_picamera_fetch(self):
        with Picamera() as picamera:
            while not self._stop_requested:
                if picamera.has_failure():
                    log.warning(picamera.error)
                    break
                image = picamera.acquire()
                if image is not None:
                    self.receive_next_sample(image=image)

    def _run_http_fetch(self, url=None, continuous=False):
        log.debug("Fetching source uri sample over http: %r", url)
        assert url
        while not self._stop_requested:
            img=None
            try:
                img = self.fetch_img(url=url)
                log.debug("""
                    Image fetched: %r
                    From URL: %r
                    """, img, url)
                log.debug('Sending sample to next element.')
                self.receive_next_sample(image=img)
            except Exception as e:
                self._on_fetch_img_exception(_exception=e)
                log.exception("""
                    Failed to fetch image from pipeline source.
                    URL: %r
                    """, url)
                if continuous:
                    log.warning("Will keep trying to fetch image from continuous source.")
                    self._fetch_img_exception_recovery()
            finally:
                if not continuous:
                    # this is not a live (continuous) media source
                    # exit the image fetch loop
                    log.debug('Completed one time http image fetch from URL: %r',
                            url)
                    break

    def _run_gst_service(self):
        log.debug("Starting Gst service process...")
        self._gst_out_queue = self._get_sample_queue()
        self._gst_process_stop_signal = multiprocessing.Event()
        self._gst_process_eos_reached = multiprocessing.Event()
        gst_service = self._get_gst_service_starter()
        self._gst_process = multiprocessing.Process(
            target=gst_service,
            name='Gstreamer Service Process',
            daemon=True,
            kwargs={'source_conf': self._source_conf,
                    'out_queue': self._gst_out_queue,
                    'stop_signal': self._gst_process_stop_signal,
                    'eos_reached': self._gst_process_eos_reached,
                    }
            )
        self._gst_process.daemon = True
        self._gst_process.start()
        gst_proc = self._gst_process
        while not self._stop_requested and gst_proc.is_alive():
            # do not use process.join() to avoid deadlock due to shared queue
            try:
                next_sample = self._gst_out_queue.get(timeout=1)
                # print('next sample received from gst queue, _on_new_sample')
                self._on_new_sample(sample=next_sample)
            except queue.Empty:
                log.debug('no new sample available yet in gst out queue')
            except Exception as e:
                log.warning('AVElement loop caught an error: %s. ',
                            str(e))
                log.warning(stacktrace())
                # print('Exception caught from _on_new_sample %r' % e)
        # print('end of _run_gst_service.')
        log.debug('exiting _run_gst_service')

    def _clear_gst_out_queue(self):
        log.debug("Clearing _gst_out_queue.")
        while not self._gst_out_queue.empty():
            try:
                self._gst_out_queue.get_nowait()
            except queue.Empty:
                log.debug("_gst_out_queue already empty.")
        log.debug("Cleared _gst_out_queue.")

    def _process_terminate(self, proc=None):
        proc.terminate()
        # give it a few seconds to terminate cleanly
        for i in range(10):
            self._clear_gst_out_queue()
            # do not use process.join() to avoid deadlock
            # due to shared queue. Use sleep instead.
            time.sleep(1)
            if not proc.is_alive():
                break

    def _process_good_kill(self, proc=None):
        # print('AVElement: Killing Gst process PID %r' % proc.pid)
        proc.kill()
        return True
        # time.sleep(3)
        # if proc.exitcode is None:
        #    # process is still alive
        #    log.warning('GST process kill was not clean. Process still alive.'
        #                'PID %r',
        #                proc.pid)
        #    # print('GST process kill was not clean. Process still alive. '
        #    #      'PID %r' %
        #    #      proc.pid)
        #    return False
        # else:
        #    log.warning('GST process killed. '
        #                'PID %r , exit code: %r',
        #                proc.pid,
        #                proc.exitcode)
        #    # print('GST process killed. '
        #    #      'PID %r , exit code: %r' %
        #    #      (proc.pid, proc.exitcode))
        #    return True

    def _stop_gst_service(self):
        log.debug("Stopping Gst service process.")
        gst_proc = self._gst_process
        stop_signal = self._gst_process_stop_signal
        if gst_proc and gst_proc.is_alive():
            # tell the OS we won't use this queue any more
            log.debug('GST process still alive. Shutting it down.')
            # log.debug('Closing out queue shared with GST proces.')
            # self._gst_out_queue.close()
            # send a polite request to the process to stop
            log.debug('Sending stop signal to GST process.')
            stop_signal.set()
            log.debug('Signalled gst process to stop')
            # give it a few seconds to stop cleanly
            for i in range(10):
                # make sure a non-empty queue doesn't block
                # the gst process from stopping
                self._clear_gst_out_queue()
                # do not use process.join() to avoid deadlock
                # due to shared queue.  Use sleep instead.
                time.sleep(1)
                if not gst_proc.is_alive():
                    break
            # process did not stop, we need to be a bit more assertive
            if gst_proc.is_alive():
                log.debug('Gst process did not stop. Terminating.')
                self._process_terminate(gst_proc)
                if gst_proc.is_alive():
                    # last resort, force kill the process
                    log.debug('Gst proess did not terminate.'
                              ' Resorting to force kill.')
                    clean_kill = self._process_good_kill(gst_proc)
                    log.debug('Gst process killed. Clean: %r.', clean_kill)
                else:
                    log.debug('Gst process stopped after terminate signal.')
            else:
                log.debug('Gst process stopped after stop signal.')

    def start(self):
        """Start processing input from the configured audio or video source."""
        super().start()
        log.info("Starting %s", self.__class__.__name__)
        self._stop_requested = False

        if self._source_conf['uri'] == "picamera":
            log.debug("Input source is picamera")
            self._run_picamera_fetch()
        elif self._source_conf['uri'].startswith('http') and \
            self._source_conf['type'] == 'image':
            log.debug("""
                Input source is an http still image: %r
                Will use python requests lib for sampling.
                """, self._source_conf['uri'])
            # use http client library to fetch still images
            self._run_http_fetch(
                url=self._source_conf['uri'],
                continuous=self._is_live)
        else:
            log.debug("""
                Input source is : %r
                Will use gstreamer for sampling.
                """, self._source_conf['uri'])
            # use gstreamer for all other types of media sources
            while not self._stop_requested:
                self._run_gst_service()
                if (self._gst_process_eos_reached and not self._is_live):
                    # gst process reached end of its input stream
                    # and this is not a live (continuous) stream loop
                    # exit the avsource element loop
                    log.debug('GST EOS reached for source uri: %r',
                            self._source_conf['uri'])
                    break
            self._stop_gst_service()
        super().stop()
        log.info("Stopped %s", self.__class__.__name__)

    def heal(self):
        """Attempt to heal a damaged AV source processing service."""
        log.debug("Entering healing method... %s", self.__class__.__name__)
        log.debug('Healing waiting for lock.')
        self._healing_in_progress.acquire()
        try:
            logging.debug('Healing lock acquired.')
            now = time.monotonic()
            # Space out healing attempts.
            # No point in back to back healing runs when there are
            # blocking dependencies on external resources.
            log.warning('latest healing ts: %r, now-MIN_HEALING_INTERVAL: %r',
                        self._latest_healing,
                        now - MIN_HEALING_INTERVAL)
            if self._latest_healing < now - MIN_HEALING_INTERVAL:
                # cause gst loop to exit and repair
                self._latest_healing = now
                self._stop_gst_service()
                # lets give external resources a chance to recover
                # for example wifi connection is down temporarily
                time.sleep(1)
                log.debug("AVElement healing completed.")
            else:
                log.debug("Healing request ignored. "
                          "Too soon after previous healing request.")
        finally:
            logging.debug('Healing lock released.')
            self._healing_in_progress.release()
        log.debug("Exiting healing method. %s", self.__class__.__name__)

    def stop(self):
        """Stop the AV source processing loop."""
        log.info("Entering stop method ... %s", self.__class__.__name__)
        self._stop_requested = True
        super().stop()
        log.info("Exiting stop method. %s", self.__class__.__name__)
