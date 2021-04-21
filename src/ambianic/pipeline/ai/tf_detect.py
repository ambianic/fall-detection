"""Tensorflow image detection wrapper."""
import logging
import time
import re
import numpy as np
# from importlib import import_module
from PIL import ImageOps
from .inference import TFInferenceEngine
from ambianic.pipeline import PipeElement


log = logging.getLogger(__name__)


class TFDetectionModel(PipeElement):
    """Applies Tensorflow image detection."""

    def __init__(self,
                 model=None,
                 labels=None,
                 label_filter=None,
                 confidence_threshold=0.6,
                 top_k=3,
                 **kwargs
                 ):
        """Initialize detector with config parameters.
        :Parameters:
        ----------
        model: ai_models/mobilenet_ssd_v2_face.tflite
        labels: ai_models/coco_labels.txt
        confidence_threshold: 0.6
        top_k: 3
        """

        # log.warning('TFImageDetection __init__ invoked')
        super().__init__(**kwargs)

        self._tfengine = TFInferenceEngine(
            model=model,
            labels=labels,
            confidence_threshold=confidence_threshold,
            top_k=top_k)
        self._labels = self.load_labels(self._tfengine.labels_path)
        self._label_filter = label_filter
        self.last_time = time.monotonic()

    def load_labels(self, label_path=None):
        """Load label mapping from integer code to text.
        :Parameters:
        ----------
        label_path : string
            Path to label mapping file.
        :Returns:
        -------
        dict
            {label_code, label_text}
        """
        assert label_path
        p = re.compile(r'\s*(\d+)(.+)')
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())
            return {int(num): text.strip() for num, text in lines}

    @staticmethod
    def thumbnail(image=None, desired_size=None):
        """Resizes original image as close as possible to desired size.
        Preserves aspect ratio of original image.
        Does not modify the original image.
        :Parameters:
        ----------
        image : PIL.Image
            Input Image for AI model detection.
        desired_size : (width, height)
            Size expected by the AI model.
        :Returns:
        -------
        PIL.Image
            Resized image fitting for the AI model input tensor.
        """
        assert image
        assert desired_size
        log.debug('input image size = %r', image.size)
        thumb = image.copy()
        w, h = desired_size
        try:
            # convert from numpy to native Python int type
            # that PIL expects
            if isinstance(w, np.generic):
                w = w.item()
                w = int(w)
                h = h.item()
                h = int(h)
            thumb.thumbnail((w, h))
        except Exception as e:
            msg = (f"Exception in "
                   f"PIL.image.thumbnail(desired_size={desired_size}):"
                   f"type(width)={type(w)}, type(height)={type(h)}"
                   f"\n{e}"
                   )
            log.exception(msg)
            raise RuntimeError(msg)
        log.debug('thmubnail image size = %r', thumb.size)
        return thumb

    @staticmethod
    def resize(image=None, desired_size=None):
        """Pad original image to exact size expected by input tensor.
        Preserve aspect ratio to avoid confusing the AI model with
        unnatural distortions. Pad the resulting image
        with solid black color pixels to fill the desired size.
        Do not modify the original image.
        :Parameters:
        ----------
        image : PIL.Image
            Input Image sized to fit an input tensor but without padding.
            Its possible that one size fits one tensor dimension exactly
            but the other size is smaller than
            the input tensor other dimension.
        desired_size : (width, height)
            Exact size expected by the AI model.
        :Returns:
        -------
        PIL.Image
            Resized image fitting exactly the AI model input tensor.
        """
        assert image
        assert desired_size
        log.debug('input image size = %r', image.size)
        thumb = image.copy()
        delta_w = desired_size[0] - thumb.size[0]
        delta_h = desired_size[1] - thumb.size[1]
        padding = (0, 0, delta_w, delta_h)
        new_im = ImageOps.expand(thumb, padding)
        log.debug('new image size = %r', new_im.size)
        assert new_im.size == desired_size
        return new_im

    @staticmethod
    def resize_to_input_tensor(image=None, desired_size=None):
        """Resize and pad original image to exact size expected by input tensor.
        Preserve aspect ratio to avoid confusing the AI model with
        distortions that it was not trained on. Pad the resulting image
        with solid black color pixels to fill the desired size.
        Do not modify anything else in the original image.
        :Parameters:
        ----------
        image : PIL.Image
            Input Image
        desired_size : (width, height)
            Exact input tensor size expected by the AI model.
        :Returns:
        -------
        PIL.Image
            Resized image fitting exactly the AI model input tensor.
        """
        assert image
        assert desired_size
        # thumbnail is a proportionately resized image
        thumbnail = TFDetectionModel.thumbnail(image=image, desired_size=desired_size)
        # convert thumbnail into an image with the exact size
        # as the input tensor
        # preserving proportions by padding as needed
        new_im = TFDetectionModel.resize(image=thumbnail, desired_size=desired_size)
        return new_im, thumbnail

    def log_stats(self, start_time=None):
        assert start_time
        log.debug("TF engine returned inference results")
        end_time = time.monotonic()
        inf_time = (end_time - start_time) * 1000
        fps = 1.0/(end_time - self.last_time)
        if self.context and self.context.unique_pipeline_name:
            pipeline_name = self.context.unique_pipeline_name
        else:
            pipeline_name = 'unknown'
        inference_type = type(self).__name__
        inf_info = '%s inference time %.2f ms, %.2f fps in pipeline %s'
        log.info(inf_info, inference_type, inf_time, fps, pipeline_name)
        self.last_time = end_time
