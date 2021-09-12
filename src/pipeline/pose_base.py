from abc import ABC, abstractmethod
import numpy as np
from PIL import ImageOps

import logging
log = logging.getLogger(__name__)

class AbstractPoseModel(ABC):
    """
        Abstract class for pose estimation models.
    """

    def __init__(self, tfengine):
                
        """Initialize posenet-base class with Tensorflow inference engine.
        :Parameters:
        ----------
        tfengine: Tensorflow inference engine.
        """

        self._tfengine = tfengine
        
        self._input_tensor_shape = self.get_input_tensor_shape()

        _, self._tensor_image_height, self._tensor_image_width, self._tensor_image_depth = \
            self.get_input_tensor_shape()

        self.confidence_threshold = self._tfengine.confidence_threshold
        log.debug(f"Initializing PoseEngine with confidence threshold \
            {self.confidence_threshold}")
        

    def get_input_tensor_shape(self):
        """Get the shape of the input tensor structure.
        Gets the shape required for the input tensor.
        For models trained for image classification / detection, the shape is
        always [1, height, width, channels].
        To be used as input for :func:`run_inference`,
        this tensor shape must be flattened into a 1-D array with size
        ``height * width * channels``. To instead get that 1-D array size, use
        :func:`required_input_array_size`.
        Returns:
        A 1-D array (:obj:`numpy.ndarray`) representing the required input
        tensor shape.
        """
        return self._tfengine.input_details[0]['shape']

 
    def tf_interpreter(self):
        return self._tfengine._tf_interpreter


    def thumbnail(self, image=None, desired_size=None):
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


    def resize(self, image=None, desired_size=None):
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


    @abstractmethod
    def execute_model(self, img):
        '''
            Execute Pose Estimation Model.
        '''