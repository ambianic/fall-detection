from src import DEFAULT_DATA_DIR
import logging
import time
from PIL import ImageDraw
from pathlib import Path


from src.pipeline.posenet_model import Posenet_MobileNet
from src.pipeline.movenet_model import Movenet

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


class PoseEngine():
    """Engine used for pose tasks."""
    def __init__(self, tfengine=None, model_name=None, context=None):
        """Creates a PoseEngine wrapper around an initialized tfengine.
        """

        assert tfengine is not None
        assert model_name is not None

        if model_name == 'movenet':
            self._model = Movenet(tfengine)
        elif model_name == 'mobilenet':
            self._model = Posenet_MobileNet(tfengine)
        
        if context:
            self._sys_data_dir = context.data_dir
        else:
            self._sys_data_dir = DEFAULT_DATA_DIR
        self._sys_data_dir = Path(self._sys_data_dir)

        self.confidence_threshold = self._model.confidence_threshold
        self._tensor_image_height = self._model._tensor_image_height
        self._tensor_image_width = self._model._tensor_image_width

        

    def draw_kps(self, kps, template_image):

        pil_im = template_image
        draw = ImageDraw.Draw(pil_im)
        
        leftShoulder = False
        rightShoulder = False
        
        scoreList = {'LShoulder_score':0,'RShoulder_score':0,'LHip_score':0,'RHip_score':0}
        
        for i in range(kps.shape[0]):
                                    
            x, y, r = int(round(kps[i, 1])), int(round(kps[i, 0])), 1

            if i == 5:
                leftShoulder = True
                leftShoulder_point = [x, y]
                scoreList['LShoulder_score'] = kps[i,-1]
                
            if i == 6:
                rightShoulder = True
                rightShoulder_point = [x, y]
                scoreList['RShoulder_score'] = kps[i,-1]

            leftUpPoint = (x-r, y-r)
            rightDownPoint = (x+r, y+r)
            twoPointList = [leftUpPoint, rightDownPoint]
            draw.ellipse(twoPointList, fill=(0, 255, 0, 255))

            if i == 11 and leftShoulder:
                leftHip_point = [x, y]
                scoreList['LHip_score'] = kps[i,-1]
                draw.line((leftShoulder_point[0],leftShoulder_point[1], leftHip_point[0],leftHip_point[1]), fill='green', width=3)

            if i == 12 and rightShoulder:
                rightHip_point = [x, y]
                scoreList['RHip_score'] = kps[i,-1]
                draw.line((rightShoulder_point[0],rightShoulder_point[1], rightHip_point[0],rightHip_point[1]), fill='green', width=3)
                    
        return pil_im, scoreList


    def get_result(self, img):

        kps, template_image, thumbnail, _inference_time = self._model.execute_model(img)
        output_img, scoreList = self.draw_kps(kps, template_image)

        return thumbnail, output_img, scoreList, _inference_time


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

        kps, template_image, thumbnail, _ = self._model.execute_model(img)
        poses = []

        keypoint_dict = {}
        cnt = 0

        keypoint_count = kps.shape[0]
        for point_i in range(keypoint_count):
            x, y = kps[point_i, 1], kps[point_i, 0]
            prob = kps[point_i, 2]

            if prob > self.confidence_threshold and \
                0 < y < self._tensor_image_height and \
                0 < x < self._tensor_image_width:

                cnt += 1
                if log.getEffectiveLevel() <= logging.DEBUG:
                    # development mode
                    # draw on image and save it for debugging
                    draw = ImageDraw.Draw(template_image)
                    draw.line(((0, 0), (x, y)), fill='blue')
            else:
                prob = 0

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
            # template_image.save(
            #                     Path(self._sys_data_dir,
            #                          debug_image_file_name),
            #                     format='JPEG')
            log.debug(f"Debug image saved: {debug_image_file_name}")
        return poses, thumbnail, pose_score
