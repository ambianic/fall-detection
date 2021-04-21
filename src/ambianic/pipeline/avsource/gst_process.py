"""Input source video processing via Gstreamer."""

import os
import sys
import threading
import signal
import logging
import traceback
from ambianic.util import stacktrace
import gi

# to prevent flake8 import reordering before setting gi versions
if 'gi' in sys.modules:
    gi.require_version('Gst', '1.0')
    gi.require_version('GstBase', '1.0')
    from gi.repository import Gst, GLib  # ,GObject,  GLib


Gst.init(None)
# No need to call GObject.threads_init() since version 3.11
# GObject.threads_init()

log = logging.getLogger(__name__)


class GstService:
    """Streams audio/video samples from various network and local A/V sources.

    Runs in a separate OS process. Reads from vadious sources and
     formatts. Serves audio/video samples in a normalized format to its master
     AVElement, which then passes on to the next element in the Ambianic
     pipeline.

    :Parameters:
    ----------
    source_conf : URI
        Source configuration. At this time URI schemes are supported such as
        rtsp://host:ip/path_to_stream.

    out_queue : multiprocessing.Queue
        The queue where this service adds samples in a normalized format
        for its master AVElement to receive and pass on to the next Ambianic
        pipeline element.

    """

    class ImageShape:
        width = height = None

    class PipelineSource:

        def __init__(self, source_conf=None):
            assert source_conf, "pipeline source configuration required."
            assert source_conf['uri'], \
                "pipeline source config missing uri element"
            # rtsp://..., rtmp://..., http://..., file:///...
            self.uri = source_conf['uri']
            # video, image, audio, auto
            self.type = source_conf.get('type', 'auto')
            self.is_live = source_conf.get('live', False)
            self.format = source_conf.get('format', None)

    def __init__(self,
                 source_conf=None,
                 out_queue=None,
                 stop_signal=None,
                 eos_reached=None):
        assert source_conf
        assert out_queue
        assert stop_signal
        assert eos_reached
        # pipeline source info
        log.debug('Initializing GstService with source: %s ', source_conf)
        self._out_queue = out_queue
        self._stop_signal = stop_signal
        self._eos_reached = eos_reached
        self.source = self.PipelineSource(source_conf=source_conf)
        # Reference to Gstreamer main loop structure
        self.mainloop = None
        # Gstreamer pipeline for a given input source
        # (could be image, audio or video)
        self.gst_pipeline = None
        self.gst_video_source = None
        self._gst_video_source_connect_id = None
        # shape of the input stream image or video
        self._source_shape = self.ImageShape()
        self.gst_queue0 = None
        self.gst_vconvert = None
        self.gst_vconvert_connect_id = None
        self.gst_queue1 = None
        # gst_appsink handlies GStreamer callbacks
        # for new media samples which it passes on to the next pipe element
        self.gst_appsink = None
        self._gst_appsink_connect_id = None
        # indicates whether stop was requested via the API
        self._stop_requested = False
        self.gst_bus = None

    def on_autoplug_continue(self, src_bin, src_pad, src_caps):
        # print('on_autoplug_continue called for uridecodebin')
        # print('src_bin: {}'.format(str(src_bin)))
        # print('src_pad: {}'.format(str(src_pad)))
        # print('src_caps: {}'.format(str(src_caps)))
        struct = src_caps.get_structure(0)
        # print("src caps struct: {}".format(struct))
        self._source_shape.width = struct["width"]
        self._source_shape.height = struct["height"]
        if self._source_shape.width:
            log.info("Input source width: %d, height: %d",
                     self._source_shape.width, self._source_shape.height)
        return True

    def _on_bus_message_eos(self, message):
        # print('GstService._handle_eos_reached')
        # if its a live source uri, we will keep trying to reconnect
        # otherwise end source input processing
        if not self.source.is_live:
            log.debug('End of stream. Exiting gstreamer loop '
                      'for this video stream.')
            self._eos_reached.set()
        self._gst_cleanup()

    def _on_bus_message_warning(self, message):
        err, debug = message.parse_warning()
        log.warning('Warning: %s: %s', err, debug)

    def _on_bus_message_error(self, message):
        err, debug = message.parse_error()
        log.warning('Error: %s: %s', err, debug)
        self._gst_cleanup()

    def _on_bus_message(self, bus, message, loop):
        t = message.type
        # print('GST: On bus message: type: %r, details: %r'
        #       % (message.type.get_name(message.type), message))
        if t == Gst.MessageType.EOS:
            self._on_bus_message_eos(message)
        elif t == Gst.MessageType.WARNING:
            self._on_bus_message_warning(message)
        elif t == Gst.MessageType.ERROR:
            self._on_bus_message_error(message)
        else:
            # pass
            log.debug('GST: Ignoring bus message: type: %r, details: %r',
                      message.type.get_name(message.type), message)
        return True

    def _on_new_sample_out_queue_full(self, sink):
        log.debug('Out queue full, skipping sample.')
        # free appsink buffer so its not blocked waiting on app pull
        sink.emit('pull-sample')
        return Gst.FlowReturn.OK

    def _on_new_sample(self, sink):
        log.debug('Input stream received new image sample.')
        if self._out_queue.full():
            return self._on_new_sample_out_queue_full(sink)
        sample = sink.emit('pull-sample')
        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        # print("gst_appsink caps struct: {}".format(struct))
        app_width = struct["width"]
        app_height = struct["height"]
        # print("gst_appsink(inference image) width: {}, height: {}".
        #   format(app_width, app_height))
        result, mapinfo = buf.map(Gst.MapFlags.READ)
        if result:
            sample = {
                'type': 'image',
                'format': 'RGB',
                'width': app_width,
                'height': app_height,
                'bytes': mapinfo.data,
            }
            log.info('GstService adding sample to out_queue.')
            self._out_queue.put(sample)
        buf.unmap(mapinfo)
        return Gst.FlowReturn.OK

    def _get_pipeline_args(self):
        log.debug('Preparing Gstreamer pipeline args')

        videosrc = self.source.uri
        videofmt = self.source.format

        if videofmt == 'h264':
            SRC_CAPS = 'video/x-h264,framerate=30/1'
        elif videofmt == 'jpeg':
            SRC_CAPS = 'image/jpeg,framerate=30/1'
        else:
            SRC_CAPS = 'video/x-raw,framerate=30/1'

        PIPELINE_SRC = "uridecodebin uri=%s use-buffering=true" % videosrc

        if videosrc.startswith('/dev/video') or videosrc.startswith('file:///dev/video'):
            PIPELINE_SRC = 'v4l2src device=%s ! %s' % (videosrc, SRC_CAPS)

        PIPELINE = """
            {pipeline_src}
             ! {leaky_q0} ! videoconvert name=vconvert ! {sink_caps}
             ! {leaky_q1} ! {sink_element}
             """

        # Ask gstreamer to format the images in a way that are close
        # to the TF model tensor.
        # Note: Having gstreamer resize doesn't appear to make
        # a big performance difference.
        # Need to look closer at hardware acceleration options where available.
        # ,width={width},pixel-aspect-ratio=1/1'
        SINK_CAPS = 'video/x-raw,format=RGB'

        LEAKY_Q_ = 'queue2 '
        LEAKY_Q0 = LEAKY_Q_ + ' name=queue0'
        LEAKY_Q1 = LEAKY_Q_ + ' name=queue1'
        
        SINK_ELEMENT = """
                appsink name=appsink sync=false
                emit-signals=true max-buffers=1 drop=true
                """
        pipeline_args = PIPELINE.format(leaky_q0=LEAKY_Q0,
                                        leaky_q1=LEAKY_Q1,
                                        sink_caps=SINK_CAPS,
                                        sink_element=SINK_ELEMENT,
                                        pipeline_src=PIPELINE_SRC)
        log.debug('Gstreamer pipeline args: %s', pipeline_args)

        print("pipeline_args ", pipeline_args)

        return pipeline_args

    def _set_gst_debug_level(self):
        if log.getEffectiveLevel() <= logging.INFO:
            # set Gst debug log level
            Gst.debug_set_active(True)
            Gst.debug_set_default_threshold(3)

    def _build_gst_pipeline(self):
        log.debug("Building new gstreamer pipeline")
        pipeline_args = self._get_pipeline_args()
        log.debug("Initializing gstreamer pipeline")
        self.gst_pipeline = Gst.parse_launch(pipeline_args)
        # self.gst_video_source = self.gst_pipeline.get_by_name('source')
        # self.gst_video_source.props.uri = self.source.uri
        # self.gst_video_source_connect_id = self.gst_video_source.connect(
        #     'autoplug-continue', self.on_autoplug_continue)
        # assert self.gst_video_source_connect_id
        self.gst_queue0 = self.gst_pipeline.get_by_name('queue0')
        self.gst_vconvert = self.gst_pipeline.get_by_name('vconvert')
        self.gst_queue1 = self.gst_pipeline.get_by_name('queue1')
        self.gst_appsink = self.gst_pipeline.get_by_name('appsink')
        log.debug("appsink: %s", str(self.gst_appsink))
        log.debug("appsink will emit signals: %s",
                  self.gst_appsink.props.emit_signals)
        # register to receive new image sample events from gst
        self._gst_appsink_connect_id = self.gst_appsink.connect(
            'new-sample', self._on_new_sample)
        self.mainloop = GLib.MainLoop()

        self._set_gst_debug_level()

        # Set up a pipeline bus watch to catch errors.
        self.gst_bus = self.gst_pipeline.get_bus()
        self.gst_bus.add_signal_watch()
        self.gst_bus.connect('message', self._on_bus_message, self.mainloop)

    def _gst_mainloop_run(self):
        log.debug("Entering main gstreamer loop")
        self.mainloop.run()
        log.debug("Exited main gstreamer loop")

    def _gst_pipeline_play(self):
        return self.gst_pipeline.set_state(Gst.State.PLAYING)

    def _gst_loop(self):
        # build new gst pipeline
        self._build_gst_pipeline()
        # Run pipeline.
        # Start playing media from source
        ret = self._gst_pipeline_play()
        # Check if the source is a live stream
        # Ref: A network-resilient example
        # https://gstreamer.freedesktop.org/documentation/tutorials/basic/streaming.html?gi-language=c
        # https://lazka.github.io/pgi-docs/Gst-1.0/classes/Pipeline.html
        # https://lazka.github.io/pgi-docs/Gst-1.0/enums.html#Gst.StateChangeReturn.NO_PREROLL
        if (ret == Gst.StateChangeReturn.FAILURE):
            raise RuntimeError('Unable to set pipeline to playing state ',
                               self.source.uri)
        elif (ret == Gst.StateChangeReturn.NO_PREROLL):
            self.source.is_live = True
            log.info("Live streaming source detected: %r", self.source.uri)
        else:
            log.debug("Gst pipeline set_state PLAYING result: %r", ret)
        self._gst_mainloop_run()

    def _gst_cleanup(self):
        log.debug("GST cleaning up resources...")
        try:
            if self.mainloop and \
              self.mainloop.is_running() and \
              self.gst_pipeline and \
              self.gst_pipeline.get_state(timeout=1)[1] != Gst.State.NULL:
                log.debug("gst pipeline still active. Terminating...")
                self.gst_pipeline.set_state(Gst.State.PAUSED)
                log.debug("self.gst_pipeline.set_state(Gst.State.PAUSED)")
                self.gst_pipeline.set_state(Gst.State.READY)
                log.debug("self.gst_pipeline.set_state(Gst.State.READY)")
                # stop pipeline elements in reverse order (from last to first)
                log.debug("gst_bus.remove_signal_watch()")
                if self.gst_bus:
                    self.gst_bus.remove_signal_watch()
                    self.gst_bus = None
                log.debug("gst_appsink.set_state(Gst.State.NULL)")
                if self.gst_appsink:
                    self.gst_appsink.set_state(Gst.State.NULL)
                    # self.gst_appsink.disconnect(self._gst_appsink_connect_id)
                    self.gst_appsink = None
                log.debug("gst_queue1.set_state(Gst.State.NULL)")
                if self.gst_queue1:
                    self.gst_queue1.set_state(Gst.State.NULL)
                    # self.gst_queue.disconnect()
                    self.gst_queue1 = None
                log.debug("gst_vconvert.set_state(Gst.State.NULL)")
                if self.gst_vconvert:
                    self.gst_vconvert.set_state(Gst.State.NULL)
                    # self.gst_vconvert.disconnect(self.gst_vconvert_connect_id)
                    self.gst_vconvert = None
                log.debug("gst_queue0.set_state(Gst.State.NULL)")
                if self.gst_queue0:
                    self.gst_queue0.set_state(Gst.State.NULL)
                    # self.gst_queue.disconnect()
                    self.gst_queue0 = None
                log.debug("gst_video_source.set_state(Gst.State.NULL)")
                if self.gst_video_source:
                    self.gst_video_source.set_state(Gst.State.NULL)
                    # self.gst_video_source.disconnect(self._gst_video_source_connect_id)
                    self.gst_video_source = None
                log.debug("gst_pipeline.set_state(Gst.State.NULL)")
                self.gst_pipeline.set_state(Gst.State.NULL)
                self.gst_pipeline = None
                log.debug("while GLib.MainContext.default().iteration(False)")
                # while GLib.MainContext.default().iteration(False):
                #     pass
            else:
                log.debug("self.gst_pipeline: %r", self.gst_pipeline)
            if self.mainloop:
                log.debug("gst mainloop.quit()")
                self.mainloop.quit()
                self.mainloop = None
            else:
                log.debug("mainloop: None")
        except Exception as e:
            log.warning('Error while cleaning up gstreamer resources: %s',
                        str(e))
            formatted_lines = traceback.format_exc().splitlines()
            log.warning('Exception stack trace: %s',
                        "\n".join(formatted_lines))
        log.debug("GST clean up exiting.")

    def _service_terminate(self, signum, frame):
        log.info('GST service caught system terminate signal %d', signum)
        if not self._stop_signal.is_set():
            self._stop_signal.set()

    def _stop_handler(self):
        self._stop_signal.wait()
        log.info('GST service received stop signal')
        self._gst_cleanup()

    def _register_stop_handler(self):
        stop_watch_thread = threading.Thread(
            name='GST stop watch thread',
            daemon=True,
            target=self._stop_handler)
        stop_watch_thread.start()

    def _register_sys_signal_handler(self):
        # Register the signal handlers
        signal.signal(signal.SIGTERM, self._service_terminate)
        signal.signal(signal.SIGINT, self._service_terminate)

    def run(self):
        """Run the gstreamer pipeline service."""
        log.info("Starting %s", self.__class__.__name__)
        self._register_sys_signal_handler()
        self._register_stop_handler()
        try:
            self._gst_loop()
        except Exception as e:
            log.warning('GST loop exited with error: %s. ',
                        str(e))
            log.warning(stacktrace())
        finally:
            log.debug('Gst service cleaning up before exit...')
            self._gst_cleanup()
            # self._out_queue.close()
            log.debug("Gst service cleaned up and ready to exit.")
        log.info("Stopped %s", self.__class__.__name__)


def start_gst_service(source_conf=None,
                      out_queue=None,
                      stop_signal=None,
                      eos_reached=None):
    svc = GstService(source_conf=source_conf,
                     out_queue=out_queue,
                     stop_signal=stop_signal,
                     eos_reached=eos_reached)
    # set priority level below parent process
    # in order to preserve UX responsiveness
    os.nice(10)
    svc.run()
    log.info('Exiting GST process')
