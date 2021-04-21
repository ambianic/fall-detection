"""Test fall detection pipe element."""
from ambianic.pipeline.ai.fall_detect import FallDetector
from ambianic import DEFAULT_DATA_DIR
from ambianic.pipeline.timeline import PipelineContext
from ambianic import logger
from pathlib import Path
from test_fall_detect import _fall_detect_config, _get_image, _OutPipeElement


_data_dir = Path(DEFAULT_DATA_DIR)
_data_dir.mkdir(parents=True, exist_ok=True)


def test_model_inputs():
    """Verify against known model inputs."""
    config = _fall_detect_config()
    fall_detector = FallDetector(**config)
    tfe = fall_detector._tfengine

    samples = tfe.input_details[0]['shape'][0]
    assert samples == 1
    height = tfe.input_details[0]['shape'][1]
    assert height == 257
    width = tfe.input_details[0]['shape'][2]
    assert width == 257
    colors = tfe.input_details[0]['shape'][3]
    assert colors == 3


def test_config_confidence_threshold():
    """Verify against known confidence threshold. Make sure it propagates
       at all levels."""
    config = _fall_detect_config()
    fall_detector = FallDetector(**config)
    tfe = fall_detector._tfengine
    pe = fall_detector._pose_engine
    assert fall_detector.confidence_threshold == config['confidence_threshold']
    assert pe.confidence_threshold == config['confidence_threshold']
    assert tfe.confidence_threshold == config['confidence_threshold']
    config['confidence_threshold'] = 0.457
    fall_detector = FallDetector(**config)
    tfe = fall_detector._tfengine
    pe = fall_detector._pose_engine
    assert fall_detector.confidence_threshold == config['confidence_threshold']
    assert pe.confidence_threshold == config['confidence_threshold']
    assert tfe.confidence_threshold == config['confidence_threshold']


def _helper_test_debug_image_save(context: PipelineContext = None):
    log_config = {
        'level': 'DEBUG'
    }
    logger.configure(config=log_config)

    # Expect to receive thumnail in result if image is provided and
    #    poses are detected.
    config = _fall_detect_config()
    result = None

    def sample_callback(image=None, thumbnail=None, inference_result=None,
                        **kwargs):
        nonlocal result
        result = image is not None and thumbnail is not None and \
            inference_result is not None

    fall_detector = FallDetector(context=context, **config)
    output = _OutPipeElement(sample_callback=sample_callback)
    fall_detector.connect_to_next_element(output)
    img_1 = _get_image(file_name='fall_img_1.png')
    fall_detector.receive_next_sample(image=img_1)
    assert result is True
    # now that we know there was a positive detection
    # lets check if the the interim debug images were saved as expected
    pose_img_files = list(_data_dir.glob('tmp-pose-detect-image*.jpg'))
    assert len(pose_img_files) > 0
    fall_img_files = list(_data_dir.glob('tmp-fall-detect-thumbnail*.jpg'))
    assert len(fall_img_files) > 0
    # cleanup after test
    all_tmp_files = pose_img_files + fall_img_files
    for f in all_tmp_files:
        f.unlink()
    # return logger level to INFO to prevent side effects in other tests
    log_config = {
        'level': 'INFO'
    }
    logger.configure(config=log_config)


def test_debug_image_save_no_context():
    """In DEBUG mode Fall detection should be saving pose and fall detection
        images while making a detection decision.
        No config context provided."""
    _helper_test_debug_image_save()


def test_debug_image_save_with_context():
    """In DEBUG mode Fall detection should be saving pose and fall detection
        images while making a detection decision. Config context provided."""
    context = PipelineContext()
    context.data_dir = DEFAULT_DATA_DIR
    _helper_test_debug_image_save(context)
