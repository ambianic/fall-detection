"""Tensorflow inference engine wrapper."""
import logging
import os
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

log = logging.getLogger(__name__)


def _get_edgetpu_interpreter(model=None):  # pragma: no cover
    # Note: Looking for ideas how to test Coral EdgeTPU dependent code
    # in a cloud CI environment such as Travis CI and Github
    tf_interpreter = None
    if model:
        try:
            edgetpu_delegate = load_delegate('libedgetpu.so.1.0')
            assert edgetpu_delegate
            tf_interpreter = Interpreter(
                model_path=model,
                experimental_delegates=[edgetpu_delegate]
                )
            log.debug('EdgeTPU available. Will use EdgeTPU model.')
        except Exception as e:
            log.debug('EdgeTPU init error: %r', e)
            # log.debug(stacktrace())
    return tf_interpreter


class TFInferenceEngine:
    """Thin wrapper around TFLite Interpreter.

    The official TFLite API is moving fast and still changes frequently.
    This class intends to abstract out underlying TF changes to some extend.

    It dynamically detects if EdgeTPU is available and uses it.
    Otherwise falls back to TFLite Runtime.
    """

    def __init__(self,
                 model=None,
                 labels=None,
                 confidence_threshold=0.8,
                 top_k=10,
                 **kwargs
                 ):
        """Create an instance of Tensorflow inference engine.

        :Parameters:
        ----------
        model: dict
            {
                'tflite': path,
                'edgetpu': path,
            }
            Where path is of type string and points to the
            location of frozen graph file (AI model).
        labels : string
            Location of file with model labels.
        confidence_threshold : float
            Inference confidence threshold.
        top_k : type
            Inference top-k threshold.

        """
        assert model
        assert model['tflite'], 'TFLite AI model path required.'
        model_tflite = model['tflite']
        assert os.path.isfile(model_tflite), \
            'TFLite AI model file does not exist: {}' \
            .format(model_tflite)
        self._model_tflite_path = model_tflite
        model_edgetpu = model.get('edgetpu', None)
        if model_edgetpu:
            assert os.path.isfile(model_edgetpu), \
                'EdgeTPU AI model file does not exist: {}' \
                .format(model_edgetpu)
        self._model_edgetpu_path = model_edgetpu
        assert labels, 'AI model labels path required.'
        assert os.path.isfile(labels), \
            'AI model labels file does not exist: {}' \
            .format(labels)
        self._model_labels_path = labels
        self._confidence_threshold = confidence_threshold
        self._top_k = top_k
        log.info('Loading AI model:\n'
                  'TFLite graph: %r\n'
                  'EdgeTPU graph: %r\n'
                  'Labels %r.'
                  'Condidence threshod: %.0f%%'
                  'top-k: %d',
                  model_tflite,
                  model_edgetpu,
                  labels,
                  confidence_threshold*100,
                  top_k)
        # EdgeTPU is not available in testing and other environments
        # load dynamically as needed
#        edgetpu_class = 'DetectionEngine'
#        module_object = import_module('edgetpu.detection.engine',
#                                      packaage=edgetpu_class)
#        target_class = getattr(module_object, edgetpu_class)
        self._tf_interpreter = _get_edgetpu_interpreter(model=model_edgetpu)
        if not self._tf_interpreter:
            log.debug('EdgeTPU not available. Will use TFLite CPU runtime.')
            self._tf_interpreter = Interpreter(model_path=model_tflite)
        assert self._tf_interpreter
        self._tf_interpreter.allocate_tensors()
        # check the type of the input tensor
        self._tf_input_details = self._tf_interpreter.get_input_details()
        self._tf_output_details = self._tf_interpreter.get_output_details()
        self._tf_is_quantized_model = \
            self.input_details[0]['dtype'] != np.float32

    @property
    def input_details(self):
        return self._tf_input_details

    @property
    def output_details(self):
        return self._tf_output_details

    @property
    def is_quantized(self):
        return self._tf_is_quantized_model

    @property
    def labels_path(self):
        """
        Location of labels file.

        :Returns:
        -------
        string
            Path to AI model labels.

        """
        return self._model_labels_path

    @property
    def confidence_threshold(self):
        """
        Inference confidence threshold.

        :Returns:
        -------
        float
            Confidence threshold for inference results.
            Only results at or above
            this threshold should be returned by each engine inference.

        """
        return self._confidence_threshold

    @property
    def top_k(self):
        """
        Inference top-k threshold.

        :Returns:
        -------
        int
            Max number of results to be returned by each inference.
            Ordered by confidence score.

        """
        return self._top_k

    def infer(self):
        """Invoke model inference on current input tensor."""
        return self._tf_interpreter.invoke()

    def set_tensor(self, index=None, tensor_data=None):
        """Set tensor data at given reference index."""
        assert isinstance(index, int)
        self._tf_interpreter.set_tensor(index, tensor_data)

    def get_tensor(self, index=None):
        """Return tensor data at given reference index."""
        assert isinstance(index, int)
        return self._tf_interpreter.get_tensor(index)
