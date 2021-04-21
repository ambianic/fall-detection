import pytest
import os

from ambianic.pipeline.ai.inference import TFInferenceEngine


def test_inference_init_no_params():
    with pytest.raises(AssertionError):
        TFInferenceEngine()


def test_inference_init_no_model_yes_labels():
    with pytest.raises(AssertionError):
        TFInferenceEngine(model=None, labels='no_data/no_labels.txt')


def test_inference_init_no_tflite_model_yes_edgemodel():
    model = {
        'edgetpu':
            'some_edgetpu.tflite'
    }
    with pytest.raises(KeyError):
        TFInferenceEngine(model=model, labels='no_data/no_labels.txt')


def _good_labels():
    _dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(_dir, 'pose_labels.txt')
    return path


def _good_tflite_model():
    _dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        _dir,
        'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
        )
    return path


def _good_edgetpu_model():
    _dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        _dir,
        'posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite'
        )
    return path


def test_inference_init_yes_tflite_model_no_edgemodel():
    model = {
        'tflite':
            _good_tflite_model(),
    }
    tf_engine = TFInferenceEngine(
        model=model,
        labels=_good_labels())
    assert tf_engine
    assert tf_engine._tf_interpreter


def test_inference_init_yes_models_yes_labels():
    model = {
        'tflite':
            _good_tflite_model(),
        'edgetpu':
            _good_edgetpu_model(),
    }
    tf_engine = TFInferenceEngine(model=model, labels=_good_labels())
    assert tf_engine
    assert tf_engine._model_tflite_path == \
        _good_tflite_model()
    assert tf_engine._model_edgetpu_path == \
        _good_edgetpu_model()


def test_inference_init_other_params():
    model = {
        'tflite':
            _good_tflite_model(),
        'edgetpu':
            _good_edgetpu_model(),
    }
    tf_engine = TFInferenceEngine(
        model=model,
        labels=_good_labels(),
        confidence_threshold=0.876,
        top_k=678,
        )
    assert tf_engine
    assert tf_engine._model_tflite_path == \
        _good_tflite_model()
    assert tf_engine._model_edgetpu_path == \
        _good_edgetpu_model()
    assert tf_engine.confidence_threshold == 0.876
    assert tf_engine.top_k == 678
    assert not tf_engine.is_quantized
    assert tf_engine._model_labels_path == _good_labels()
