from src import DEFAULT_DATA_DIR
import logging
import time
import numpy as np
from PIL import ImageDraw
from pathlib import Path
from PIL import ImageOps

log = logging.getLogger(__name__)

KEYPOINTS = (
  'nose',
  'left eye',
  'right eye',
  'left ear',
  'right ear',
  'left shoulder',
  'right shoulder',
  'left elbow',
  'right elbow',
  'left wrist',
  'right wrist',
  'left hip',
  'right hip',
  'left knee',
  'right knee',
  'left ankle',
  'right ankle'
)


class Keypoint:
    __slots__ = ['k', 'yx', 'score']

    def __init__(self, k, yx, score=None):
        self.k = k
        self.yx = yx
        self.score = score

    def __repr__(self):
        return 'Keypoint(<{}>, {}, {})'.format(self.k, self.yx, self.score)


class Pose:
    __slots__ = ['keypoints', 'score']

    def __init__(self, keypoints, score=None):
        assert len(keypoints) == len(KEYPOINTS)
        self.keypoints = keypoints
        self.score = score

    def __repr__(self):
        return 'Pose({}, {})'.format(self.keypoints, self.score)


class PoseEngine:
    """Engine used for pose tasks."""
    def __init__(self, tfengine=None, context=None):
        """Creates a PoseEngine wrapper around an initialized tfengine.
        """
        if context:
            self._sys_data_dir = context.data_dir
        else:
            self._sys_data_dir = DEFAULT_DATA_DIR
        self._sys_data_dir = Path(self._sys_data_dir)
        assert tfengine is not None
        self._tfengine = tfengine

        self._input_tensor_shape = self.get_input_tensor_shape()

        _, self._tensor_image_height, self._tensor_image_width, self._tensor_image_depth = \
            self.get_input_tensor_shape()

        self.confidence_threshold = self._tfengine.confidence_threshold
        self.confidence_threshold = 0.11
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


    def parse_output(self, keypoints_with_scores, keypoint_threshold=0.11):
  
        keypoints_all = []
        num_instances, _, _, _ = keypoints_with_scores.shape
        
        for idx in range(num_instances):
            
            kpts_y = keypoints_with_scores[0, idx, :, 1]
            kpts_x = keypoints_with_scores[0, idx, :, 0]
            
            kpts_scores = keypoints_with_scores[0, idx, :, 2]
            
            kpts_absolute_xy = np.stack([self._tensor_image_width * np.array(kpts_x), self._tensor_image_height * np.array(kpts_y), kpts_scores], axis=-1)
            keypoints_all.append(kpts_absolute_xy)

        keypoints_xy = np.concatenate(keypoints_all, axis=0)
        
        return keypoints_xy


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


    def detect_poses(self, img):
        """
        Detects poses in a given image.
        :Parameters:
        ----------
        img : PIL.Image
            Input Image for AI model detection.
        :Returns:
        -------
        poses:
            A list of Pose objects with keypoints and confidence scores
        PIL.Image
            Resized image fitting the AI model input tensor.
        """

        _tensor_input_size = (self._tensor_image_width,
                              self._tensor_image_height)

        # thumbnail is a proportionately resized image
        thumbnail = self.thumbnail(image=img,
                                               desired_size=_tensor_input_size)
        # convert thumbnail into an image with the exact size
        # as the input tensor preserving proportions by padding with
        # a solid color as needed
        template_image = self.resize(image=thumbnail,
                                desired_size=_tensor_input_size)

        template_input = np.expand_dims(template_image.copy(), axis=0)
        floating_model = self._tfengine.input_details[0]['dtype'] == np.float32

        if floating_model:
            template_input = template_input.astype(np.float32)

        self.tf_interpreter().\
            set_tensor(self._tfengine.input_details[0]['index'],
                       template_input)
        self.tf_interpreter().invoke()

        keypoints_with_scores = self.tf_interpreter().\
            get_tensor(self._tfengine.output_details[0]['index'])

        kps = self.parse_output(keypoints_with_scores)
        
        poses = []

        keypoint_dict = {}
        cnt = 0

        keypoint_count = kps.shape[0]
        for point_i in range(keypoint_count):
            x, y = kps[point_i, 1], kps[point_i, 0]
            prob = kps[point_i, 2]

            if prob > self.confidence_threshold:
                cnt += 1
                if log.getEffectiveLevel() <= logging.DEBUG:
                    # development mode
                    # draw on image and save it for debugging
                    draw = ImageDraw.Draw(template_image)
                    draw.line(((0, 0), (x, y)), fill='blue')

            keypoint = Keypoint(KEYPOINTS[point_i], [x, y], prob)
            keypoint_dict[KEYPOINTS[point_i]] = keypoint

        # overall pose score is calculated as the average of all
        # individual keypoint scores
        pose_score = cnt/keypoint_count
        log.debug(f"Overall pose score (keypoint score average): {pose_score}")
        poses.append(Pose(keypoint_dict, pose_score))
        if cnt > 0 and log.getEffectiveLevel() <= logging.DEBUG:
            # development mode
            # save template_image for debugging
            timestr = int(time.monotonic()*1000)
            log.debug(f"Detected a pose with {cnt} keypoints that score over \
                the minimum confidence threshold of \
                {self.confidence_threshold}.")
            debug_image_file_name = \
                f'tmp-pose-detect-image-time-{timestr}-keypoints-{cnt}.jpg'
            template_image.save(
                                Path(self._sys_data_dir,
                                     debug_image_file_name),
                                format='JPEG')
            log.debug(f"Debug image saved: {debug_image_file_name}")
        return poses, thumbnail, pose_score
