"""Test audio/video source pipeline element."""
import pytest
from ambianic.pipeline.avsource.av_element \
    import AVSourceElement, MIN_HEALING_INTERVAL
import threading
import os
import pathlib
import time
from ambianic.pipeline import PipeElement
from ambianic.pipeline.ai.fall_detect import FallDetector
from ambianic.pipeline.avsource.gst_process import GstService
import logging

from test_avsource_picamera import picamera_override
from ambianic.pipeline.avsource import picam

log = logging.getLogger()
log.setLevel(logging.DEBUG)


class _TestAVSourceElement(AVSourceElement):

    def __init__(self, **source_conf):
        super().__init__(**source_conf)
        self._run_gst_service_called = False
        self._stop_gst_service_called = False

    def _run_gst_service(self):
        self._run_gst_service_called = True

    def _stop_gst_service(self):
        self._stop_gst_service_called = True


class _OutPipeElement(PipeElement):

    def __init__(self, sample_callback=None):
        super().__init__()
        assert sample_callback
        self._sample_callback = sample_callback

    def receive_next_sample(self, **sample):
        self._sample_callback(**sample)


def test_no_config():
    with pytest.raises(AssertionError):
        AVSourceElement()


def test_start_stop_dummy_source():
    avsource = _TestAVSourceElement(uri='rstp://blah', type='video')
    t = threading.Thread(
        name="Test AVSourceElement",
        target=avsource.start, daemon=True
        )
    t.start()
    t.join(timeout=1)
    assert avsource._run_gst_service_called
    assert t.is_alive()
    avsource.stop()
    t.join(timeout=1)
    assert avsource._stop_gst_service_called
    assert not t.is_alive()


def test_start_stop_file_source_image_size():
    """Expect to receive an image with dimentions of the input video frame."""
    _dir = os.path.dirname(os.path.abspath(__file__))
    video_file = os.path.join(
        _dir,
        'test_images_video.avi'
        )
    abs_path = os.path.abspath(video_file)
    video_uri = pathlib.Path(abs_path).as_uri()
    avsource = AVSourceElement(uri=video_uri, type='video')
    sample_received = threading.Event()
    sample_image = None

    def sample_callback(image=None, inference_result=None, **kwargs):
        nonlocal sample_image
        nonlocal sample_received
        sample_image = image
        sample_received.set()
    output = _OutPipeElement(sample_callback=sample_callback)
    avsource.connect_to_next_element(output)
    t = threading.Thread(
        name="Test AVSourceElement",
        target=avsource.start, daemon=True
        )
    t.start()
    sample_received.wait(timeout=5)
    assert sample_image
    assert sample_image.size[0] == 640
    assert sample_image.size[1] == 480
    assert t.is_alive()
    avsource.stop()
    t.join(timeout=20)
    assert not t.is_alive()


def _fall_detect_config():

    _dir = os.path.dirname(os.path.abspath(__file__))
    _good_tflite_model = os.path.join(
        _dir,
        '../ai/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
        )
    _good_edgetpu_model = os.path.join(
        _dir,
        '../ai/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite'
        )
    _good_labels = os.path.join(_dir, '../ai/pose_labels.txt')
    config = {
        'model': {
            'tflite': _good_tflite_model,
            'edgetpu': _good_edgetpu_model,
            },
        'labels': _good_labels,
        'top_k': 3,
        'confidence_threshold': 0.6,
    }
    return config


def test_start_stop_file_source_fall_detect():
    """Expect to detect a fall in the video sample."""
    _dir = os.path.dirname(os.path.abspath(__file__))
    video_file = os.path.join(
        _dir,
        'test_images_video.avi'
        )
    abs_path = os.path.abspath(video_file)
    video_uri = pathlib.Path(abs_path).as_uri()
    avsource = AVSourceElement(uri=video_uri, type='video')
    config = _fall_detect_config()
    detection_received = threading.Event()
    sample_image = None
    detections = None

    def sample_callback(image=None, inference_result=None, **kwargs):
        nonlocal sample_image
        nonlocal detection_received
        sample_image = image
        nonlocal detections
        detections = inference_result
        print('detections: {det}'.format(det=detections))
        print('len(detections): {len}'.format(len=len(detections)))
        if detections:
            label = detections[0]['label']
            confidence = detections[0]['confidence']

            if label == 'FALL' and confidence > 0.6:
                # skip video image samples until we reach a fall detection
                # with high level of confidence
                detection_received.set()
    fall_detector = FallDetector(**config)
    fall_detector.min_time_between_frames = 0.05
    avsource.connect_to_next_element(fall_detector)
    output = _OutPipeElement(sample_callback=sample_callback)
    fall_detector.connect_to_next_element(output)
    t = threading.Thread(
        name="Test AVSourceElement",
        target=avsource.start, daemon=True
        )
    t.start()
    detection_received.wait(timeout=2)
    assert sample_image
    assert sample_image.size[0] == 640
    assert sample_image.size[1] == 480
    assert detections
    assert len(detections) == 1

    label = detections[0]['label']
    confidence = detections[0]['confidence']
    angle = detections[0]['leaning_angle']
    keypoint_corr = detections[0]['keypoint_corr']

    assert label == 'FALL'
    assert confidence > 0.6
    assert keypoint_corr
    assert angle > 60
    avsource.stop()
    t.join(timeout=10)
    assert not t.is_alive()


def test_stop_on_video_EOS():
    """Processing should stop when AVSource reaches end of input stream."""
    _dir = os.path.dirname(os.path.abspath(__file__))
    video_file = os.path.join(
        _dir,
        'test_images_video.avi'
        )
    abs_path = os.path.abspath(video_file)
    video_uri = pathlib.Path(abs_path).as_uri()
    avsource = AVSourceElement(uri=video_uri, type='video')
    sample_received = threading.Event()
    sample_image = None

    def sample_callback(image=None, inference_result=None, **kwargs):
        nonlocal sample_image
        nonlocal sample_received
        sample_image = image
        sample_received.set()
    output = _OutPipeElement(sample_callback=sample_callback)
    avsource.connect_to_next_element(output)
    t = threading.Thread(
        name="Test AVSourceElement",
        target=avsource.start, daemon=True
        )
    t.start()
    sample_received.wait(timeout=5)
    assert sample_image
    assert sample_image.size[0] == 640
    assert sample_image.size[1] == 480
    assert t.is_alive()
    avsource._gst_process_eos_reached.wait(timeout=30)
    if not avsource._gst_process_eos_reached.is_set():
        # Intermitently gstreamer does not feed an EOS message
        # on the event bus when it reaches end of the video file.
        # This is a known issue under investigation.
        # Let's send a stop signal to the pipeline.
        avsource.stop()
    t.join(timeout=10)
    assert not t.is_alive()


def test_picamera_fail_import():
    # mock picamera module
    picam.picamera_override = None

    avsource = AVSourceElement(uri="picamera", type='video')
    t = threading.Thread(
        name="Test AVSourceElement Picamera",
        target=avsource.start, daemon=True
    )
    t.start()
    time.sleep(1)
    t.join(timeout=10)
    assert not t.is_alive()


def test_picamera_input():
    # mock picamera module
    picam.picamera_override = picamera_override

    avsource = AVSourceElement(uri="picamera", type='video')
    config = _fall_detect_config()
    detection_received = threading.Event()
    sample_image = None
    detections = None

    def sample_callback(image=None, inference_result=None, **kwargs):
        nonlocal sample_image
        nonlocal detection_received
        sample_image = image
        nonlocal detections
        detections = inference_result
        print('detections: {det}'.format(det=detections))
        print('len(detections): {len}'.format(len=len(detections)))
        if detections:
            label = detections[0]['label']
            confidence = detections[0]['confidence']
            if label == 'FALL' and confidence > 0.6:
                # skip video image samples until we reach a fall detection
                # with high level of confidence
                detection_received.set()
    fall_detector = FallDetector(**config)
    avsource.connect_to_next_element(fall_detector)
    output = _OutPipeElement(sample_callback=sample_callback)
    fall_detector.connect_to_next_element(output)
    t = threading.Thread(
        name="Test AVSourceElement",
        target=avsource.start, daemon=True
    )
    t.start()
    detection_received.wait(timeout=10)
    assert sample_image
    assert sample_image.size[0] == 1280
    assert sample_image.size[1] == 720
    assert detections
    assert len(detections) == 1
    label = detections[0]['label']
    confidence = detections[0]['confidence']
    angle = detections[0]['leaning_angle']
    keypoint_corr = detections[0]['keypoint_corr']

    assert label == 'FALL'
    assert confidence > 0.6
    assert keypoint_corr
    assert angle > 60
    avsource.stop()
    t.join(timeout=10)
    assert not t.is_alive()


# def test_picamera_input_fail():
#     # mock picamera module
#     picam.picamera_override = picamera_override_failure
#     avsource = AVSourceElement(uri="picamera", type='video')
#     t = threading.Thread(
#         name="Test AVSourceElement",
#         target=avsource.start, daemon=True
#     )
#     t.start()
#     time.sleep(1)
#     avsource.stop()
#     t.join(timeout=10)
#     assert not t.is_alive()


def test_heal():
    avsource = _TestAVSourceElement(uri='rstp://blah', type='image')
    t = threading.Thread(
        name="Test AVSourceElement",
        target=avsource.start, daemon=True
        )
    t.start()
    t.join(timeout=1)
    # simulate the pipe element has been unhealthy for long enough
    avsource._latest_healing = \
        avsource._latest_healing - 2*MIN_HEALING_INTERVAL
    avsource.heal()
    # heal should have done its job and stopped the gst service for repair
    assert avsource._stop_gst_service_called
    latest = avsource._latest_healing
    avsource.heal()
    # heal should ignore back to back requests
    # latest healing timestamp should be unchanged
    assert latest == avsource._latest_healing
    # set the latest healing clock back by more than the min interval
    avsource._latest_healing = \
        avsource._latest_healing - 2*MIN_HEALING_INTERVAL
    avsource.heal()
    # now the healing process should have ran and
    # set the latest timestamp to a more recent time than the last healing run
    assert latest < avsource._latest_healing
    assert t.is_alive()
    # reset the gst stop flag to test if
    # the method will be called again by stop()
    avsource._stop_gst_service_called = False
    avsource.stop()
    t.join(timeout=1)
    assert avsource._stop_gst_service_called
    assert not t.is_alive()


class _TestAVSourceElement2(AVSourceElement):

    def __init__(self, **source_conf):
        super().__init__(**source_conf)
        self._bad_sample_processed_re = False
        self._bad_sample_processed_ae = False

    def _get_sample_queue(self):
        q = super()._get_sample_queue()
        # put a fake bad sample on the queue to test exception handling
        q.put('A bad sample to test RuntimeError.')
        q.put('Another bad sample to test AssertionError.')
        return q

    def _on_new_sample(self, sample=None):
        if not self._bad_sample_processed_re:
            # through a RuntimeError once then proceed as normal
            # the pipe element should log the exception but keep running
            print('RuntimeError during processing.')
            self._bad_sample_processed_re = True
            raise RuntimeError('Something went wrong during processing.')
        if not self._bad_sample_processed_ae:
            # through an AssertionError once then proceed as normal
            # the pipe element should log the exception but keep running
            print('AssertionError during processing.')
            self._bad_sample_processed_ae = True
            raise AssertionError('Something went wrong during processing.')
        super()._on_new_sample(sample)


def test_exception_on_new_sample():
    """Exception from _on_new_sample() should not break the pipe loop."""
    _dir = os.path.dirname(os.path.abspath(__file__))
    video_file = os.path.join(
        _dir,
        'test_images_video.avi'
        )
    abs_path = os.path.abspath(video_file)
    video_uri = pathlib.Path(abs_path).as_uri()
    avsource = _TestAVSourceElement2(uri=video_uri, type='image')
    config = _fall_detect_config()
    detection_received = threading.Event()
    sample_image = None
    detections = None

    def sample_callback(image=None, inference_result=None, **kwargs):
        nonlocal sample_image
        nonlocal detection_received
        sample_image = image
        nonlocal detections
        detections = inference_result
        print('detections: {det}'.format(det=detections))
        print('len(detections): {len}'.format(len=len(detections)))
        if detections:
            label = detections[0]['label']
            confidence = detections[0]['confidence']
            if label == 'FALL' and confidence > 0.6:
                # skip video image samples until we reach a person detection
                # with high level of confidence
                detection_received.set()
    fall_detector = FallDetector(**config)
    fall_detector.min_time_between_frames = 0.05
    avsource.connect_to_next_element(fall_detector)
    output = _OutPipeElement(sample_callback=sample_callback)
    fall_detector.connect_to_next_element(output)
    t = threading.Thread(
        name="Test AVSourceElement",
        target=avsource.start, daemon=True
        )
    t.start()
    detection_received.wait(timeout=10)
    assert sample_image
    assert sample_image.size[0] == 640
    assert sample_image.size[1] == 480
    assert detections
    assert len(detections) == 1
    label = detections[0]['label']
    confidence = detections[0]['confidence']
    angle = detections[0]['leaning_angle']
    keypoint_corr = detections[0]['keypoint_corr']
    assert label == 'FALL'
    assert confidence > 0.6
    assert keypoint_corr
    assert angle > 60
    avsource._gst_process_eos_reached.wait(timeout=5)
    assert avsource._gst_process_eos_reached.is_set()
    t.join(timeout=3)
    assert not t.is_alive()


def _test_start_gst_service3(source_conf=None,
                             out_queue=None,
                             stop_signal=None,
                             eos_reached=None):
    print('_test_start_gst_service3 returning _TestGstService3')
    svc = _TestGstService3(source_conf=source_conf,
                           out_queue=out_queue,
                           stop_signal=stop_signal,
                           eos_reached=eos_reached)
    svc.run()
    print('Exiting GST process')


class _TestGstService3(GstService):

    def _stop_handler(self):
        self._stop_signal.wait()
        print('_TestGstService3 service received stop signal')
        # Don't stop gracefully to force kill test
        # self._gst_cleanup()

    # simulate endless stream processing to force
    # process kill test
    def _on_bus_message_eos(self, message):
        print('_TestGstService3._on_bus_message_eos ignoring EOS signal.')


class _TestAVSourceElement3(AVSourceElement):

    def __init__(self, **source_conf):
        super().__init__(**source_conf)
        self._clean_kill = False

    def _get_gst_service_starter(self):
        print('_TestAVSourceElement3._get_gst_service_starter returning '
              ' _test_start_gst_service')
        return _test_start_gst_service3

    def _process_good_kill(self, proc=None):
        print('_TestAVSourceElement3: Killing Gst process PID %r' % proc.pid)
        self._clean_kill = super()._process_good_kill(proc)
        print('_TestAVSourceElement3: Gst process killed cleanly: %r'
              % self._clean_kill)
        return self._clean_kill


def test_gst_process_kill():
    """Gst process kill when it doesn't respond to stop and terminate."""
    _dir = os.path.dirname(os.path.abspath(__file__))
    video_file = os.path.join(
        _dir,
        'test_images_video.avi'
        )
    abs_path = os.path.abspath(video_file)
    video_uri = pathlib.Path(abs_path).as_uri()
    avsource = _TestAVSourceElement3(uri=video_uri, type='image')
    config = _fall_detect_config()
    detection_received = threading.Event()
    sample_image = None
    detections = None

    def sample_callback(image=None, inference_result=None, **kwargs):
        nonlocal sample_image
        nonlocal detection_received
        sample_image = image
        nonlocal detections
        detections = inference_result
        print('detections: {det}'.format(det=detections))
        print('len(detections): {len}'.format(len=len(detections)))
        if detections:
            label = detections[0]['label']
            confidence = detections[0]['confidence']
            if label == 'FALL' and confidence > 0.6:
                # skip video image samples until we reach a fall detection
                # with high level of confidence
                detection_received.set()
    fall_detector = FallDetector(**config)
    fall_detector.min_time_between_frames = 0.05
    avsource.connect_to_next_element(fall_detector)
    output = _OutPipeElement(sample_callback=sample_callback)
    fall_detector.connect_to_next_element(output)
    t = threading.Thread(
        name="Test AVSourceElement",
        target=avsource.start, daemon=True
        )
    t.start()
    detection_received.wait(timeout=10)
    assert sample_image
    assert sample_image.size[0] == 640
    assert sample_image.size[1] == 480
    assert detections
    assert len(detections) == 1
    label = detections[0]['label']
    confidence = detections[0]['confidence']
    angle = detections[0]['leaning_angle']
    keypoint_corr = detections[0]['keypoint_corr']
    assert label == 'FALL'
    assert confidence > 0.6
    assert keypoint_corr
    assert angle > 60
    avsource.stop()
    t.join(timeout=30)
    assert not t.is_alive()
    assert avsource._clean_kill


class _TestGstService4(GstService):

    _terminate_called = False

    def _service_terminate(self, signum, frame):
        print('_TestGstService4 service caught system terminate signal %d'
              % signum)
        self._terminate_called = True
        super()._service_terminate(signum, frame)

    def _stop_handler(self):
        print('_TestGstService4 service received stop signal')
        while not self._terminate_called:
            time.sleep(0.5)
        # ignore stop signals to force process.terminate() call
        super()._stop_handler()

    # simulate endless stream processing to force
    # process terminate test
    def _on_bus_message_eos(self, message):
        print('_TestGstService4._on_bus_message_eos ignoring EOS signal.')


def _test_start_gst_service4(source_conf=None,
                             out_queue=None,
                             stop_signal=None,
                             eos_reached=None):
    print('_test_start_gst_service returning _TestGstService')
    svc = _TestGstService4(source_conf=source_conf,
                           out_queue=out_queue,
                           stop_signal=stop_signal,
                           eos_reached=eos_reached)
    svc.run()
    print('_test_start_gst_service4: Exiting GST process')


class _TestAVSourceElement4(AVSourceElement):

    def __init__(self, **source_conf):
        super().__init__(**source_conf)
        self._terminate_requested = False
        self._clean_terminate = True

    def _get_gst_service_starter(self):
        print('_TestAVSourceElement4._get_gst_service_starter '
              'returning _test_start_gst_service4')
        return _test_start_gst_service4

    def _process_terminate(self, proc=None):
        self._terminate_requested = True
        super()._process_terminate(proc)

    def _process_good_kill(self, proc=None):
        # kill step should not be reached if terminate worked
        self._clean_terminate = False
        print('_TestAVSourceElement4: Kill terminate should not be reached '
              'when terminate worked.')
        clean_kill = super()._process_good_kill(proc)
        return clean_kill


def test_gst_process_terminate():
    """Gst process terminate when it doesn't respond to stop signal."""
    _dir = os.path.dirname(os.path.abspath(__file__))
    video_file = os.path.join(
        _dir,
        'test_images_video.avi'
        )
    abs_path = os.path.abspath(video_file)
    video_uri = pathlib.Path(abs_path).as_uri()
    avsource = _TestAVSourceElement4(uri=video_uri, type='image')
    config = _fall_detect_config()
    detection_received = threading.Event()
    sample_image = None
    detections = None

    def sample_callback(image=None, inference_result=None, **kwargs):
        nonlocal sample_image
        nonlocal detection_received
        sample_image = image
        nonlocal detections
        detections = inference_result
        print('detections: {det}'.format(det=detections))
        print('len(detections): {len}'.format(len=len(detections)))
        if detections:
            label = detections[0]['label']
            confidence = detections[0]['confidence']
            if label == 'FALL' and confidence > 0.6:
                # skip video image samples until we reach a fall detection
                # with high level of confidence
                detection_received.set()
    fall_detector = FallDetector(**config)
    fall_detector.min_time_between_frames = 0.05
    avsource.connect_to_next_element(fall_detector)
    output = _OutPipeElement(sample_callback=sample_callback)
    fall_detector.connect_to_next_element(output)
    t = threading.Thread(
        name="Test AVSourceElement",
        target=avsource.start, daemon=True
        )
    t.start()
    detection_received.wait(timeout=5)
    assert sample_image
    assert sample_image.size[0] == 640
    assert sample_image.size[1] == 480
    assert detections
    assert len(detections) == 1
    label = detections[0]['label']
    confidence = detections[0]['confidence']
    angle = detections[0]['leaning_angle']
    keypoint_corr = detections[0]['keypoint_corr']
    assert label == 'FALL'
    assert confidence > 0.6
    assert keypoint_corr
    assert angle > 60
    avsource.stop()
    t.join(timeout=30)
    assert not t.is_alive()
    assert avsource._terminate_requested
    assert avsource._clean_terminate
