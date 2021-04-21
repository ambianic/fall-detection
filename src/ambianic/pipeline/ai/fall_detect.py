"""Fall detection pipe element."""
from ambianic.pipeline.ai.tf_detect import TFDetectionModel
from ambianic.pipeline.ai.pose_engine import PoseEngine
from ambianic import DEFAULT_DATA_DIR
import logging
import math
import time
from PIL import Image, ImageDraw
from pathlib import Path

log = logging.getLogger(__name__)


class FallDetector(TFDetectionModel):

    """Detects falls comparing two images spaced about 1-2 seconds apart."""
    def __init__(self,
                 model=None,
                 confidence_threshold=0.15,
                 **kwargs
                 ):
        """Initialize detector with config parameters.
        :Parameters:
        ----------
        model: dict
        {
            'tflite': 
                'ai_models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
            'edgetpu': 
                'ai_models/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite'
        }
        """
        super().__init__(model=model,
                         confidence_threshold=confidence_threshold,
                         **kwargs)

        if self.context:
            self._sys_data_dir = self.context.data_dir
        else:
            self._sys_data_dir = DEFAULT_DATA_DIR
        self._sys_data_dir = Path(self._sys_data_dir)

        # previous pose detection information for frame at time t-1 and t-2 \
        # to compare pose changes against
        self._prev_data = [None] * 2

        # Data of previous frames lookup constants
        self.POSE_VAL = '_prev_pose_dix'
        self.TIMESTAMP = '_prev_time'
        self.THUMBNAIL = '_prev_thumbnail'
        self.LEFT_ANGLE_WITH_YAXIS = '_prev_left_angle_with_yaxis'
        self.RIGHT_ANGLE_WITH_YAXIS = '_prev_right_angle_with_yaxis'
        self.BODY_VECTOR_SCORE = '_prev_body_vector_score'

        _dix = {self.POSE_VAL: [],
                self.TIMESTAMP: time.monotonic(),
                self.THUMBNAIL: None,
                self.LEFT_ANGLE_WITH_YAXIS: None,
                self.RIGHT_ANGLE_WITH_YAXIS: None,
                self.BODY_VECTOR_SCORE: 0
                }

        # self._prev_data[0] : store data of frame at t-2
        # self._prev_data[1] : store data of frame at t-1
        self._prev_data[0] = self._prev_data[1] = _dix

        self._pose_engine = PoseEngine(self._tfengine, context=self.context)
        self._fall_factor = 60
        self.confidence_threshold = confidence_threshold
        log.debug(f"Initializing FallDetector with conficence threshold: \
                  {self.confidence_threshold}")

        # Require a minimum amount of time between two video frames in seconds.
        # Otherwise on high performing hard, the poses could be too close to
        # each other and have negligible difference
        # for fall detection purpose.
        self.min_time_between_frames = 1
        # Require the time distance between two video frames not to exceed
        # a certain limit in seconds.
        # Otherwise there could be data noise which could lead
        # false positive detections.
        self.max_time_between_frames = 10

        # keypoint lookup constants
        self.LEFT_SHOULDER = 'left shoulder'
        self.LEFT_HIP = 'left hip'
        self.RIGHT_SHOULDER = 'right shoulder'
        self.RIGHT_HIP = 'right hip'

        self.fall_detect_corr = [self.LEFT_SHOULDER, self.LEFT_HIP,
                                 self.RIGHT_SHOULDER, self.RIGHT_HIP]

    def process_sample(self, **sample):
        """Detect objects in sample image."""
        log.debug("%s received new sample", self.__class__.__name__)
        if not sample:
            # pass through empty samples to next element
            yield None
        else:
            try:
                image = sample['image']
                inference_result, thumbnail = self.fall_detect(image=image)
                inference_result = self.convert_inference_result(
                                        inference_result)
                inf_meta = {
                    'display': 'Fall Detection',
                }
                # pass on the results to the next connected pipe element
                processed_sample = {
                    'image': image,
                    'thumbnail': thumbnail,
                    'inference_result': inference_result,
                    'inference_meta': inf_meta
                    }
                yield processed_sample
            except Exception as e:
                log.exception('Error "%s" while processing sample. '
                              'Dropping sample: %s',
                              str(e),
                              str(sample))

    def calculate_angle(self, p):
        '''
            Calculate angle b/w two lines such as
            left shoulder-hip with prev frame's left shoulder-hip or
            right shoulder-hip with prev frame's right shoulder-hip or
            shoulder-hip line with vertical axis
        '''

        x1, y1 = p[0][0]
        x2, y2 = p[0][1]

        x3, y3 = p[1][0]
        x4, y4 = p[1][1]

        theta1 = math.degrees(math.atan2(y2-y1, x1-x2))
        theta2 = math.degrees(math.atan2(y4-y3, x3-x4))

        angle = abs(theta1-theta2)
        return angle

    def is_body_line_motion_downward(self, left_angle_with_yaxis,
                                     rigth_angle_with_yaxis, inx):

        test = False

        l_angle = left_angle_with_yaxis and \
            self._prev_data[inx][self.LEFT_ANGLE_WITH_YAXIS]  \
            and left_angle_with_yaxis > \
            self._prev_data[inx][self.LEFT_ANGLE_WITH_YAXIS]
        r_angle = rigth_angle_with_yaxis and \
            self._prev_data[inx][self.RIGHT_ANGLE_WITH_YAXIS] \
            and rigth_angle_with_yaxis > \
            self._prev_data[inx][self.RIGHT_ANGLE_WITH_YAXIS]

        if l_angle or r_angle:
            test = True

        return test

    def find_keypoints(self, image):

        # this score value should be related to the configuration \
        # confidence_threshold parameter
        min_score = self.confidence_threshold
        rotations = [Image.ROTATE_270, Image.ROTATE_90]
        angle = 0
        pose = None
        poses, thumbnail, _ = self._pose_engine.detect_poses(image)
        width, height = thumbnail.size
        # if no pose detected with high confidence,
        # try rotating the image +/- 90' to find a fallen person
        # currently only looking at pose[0] because we are focused \
        # on a lone person falls
        # while (not poses or poses[0].score < min_score) and rotations:
        spinal_vector_score, pose_dix = self.estimate_spinal_vector_score(
                                        poses[0])
        while spinal_vector_score < min_score and rotations:
            angle = rotations.pop()
            transposed = image.transpose(angle)
            # we are interested in the poses but not the rotated thumbnail
            poses, _, _ = self._pose_engine.detect_poses(transposed)
            spinal_vector_score, pose_dix = self.estimate_spinal_vector_score(
                                    poses[0])

        if poses and poses[0]:
            pose = poses[0]

        # lets check if we found a good pose candidate

        if (pose and spinal_vector_score >= min_score):
            # if the image was rotated, we need to rotate back to the original\
            # image coordinates
            # before comparing with poses in other frames.
            if angle == Image.ROTATE_90:
                # ROTATE_90 rotates 90' counter clockwise \
                # from ^ to < orientation.
                for _, keypoint in pose.keypoints.items():
                    # keypoint.yx[0] is the x coordinate in an image
                    # keypoint.yx[0] is the y coordinate in an image, \
                    # with 0,0 in the upper left corner (not lower left).
                    tmp_swap = keypoint.yx[0]
                    keypoint.yx[0] = width-keypoint.yx[1]
                    keypoint.yx[1] = tmp_swap
            elif angle == Image.ROTATE_270:
                # ROTATE_270 rotates 90' clockwise from ^ to > orientation.
                for _, keypoint in pose.keypoints.items():
                    tmp_swap = keypoint.yx[0]
                    keypoint.yx[0] = keypoint.yx[1]
                    keypoint.yx[1] = height-tmp_swap
            # we could not detexct a pose with sufficient confidence
            log.info(f"""A pose detected with
                    spinal_vector_score={spinal_vector_score} >= {min_score}
                    confidence threshold.
                    Pose keypoints: {pose_dix}"
                """)
        else:
            pose = None

        return pose, thumbnail, spinal_vector_score, pose_dix

    def find_changes_in_angle(self, pose_dix, inx):
        '''
            Find the changes in angle for shoulder-hip lines
            b/w current and previpus frame.
        '''

        prev_leftLine_corr_exist = all(e in self._prev_data[inx][self.POSE_VAL]
                                       for e in [self.LEFT_SHOULDER, self.LEFT_HIP])
        curr_leftLine_corr_exist = all(e in pose_dix for e in [self.LEFT_SHOULDER,self.LEFT_HIP])

        prev_rightLine_corr_exist = all(e in self._prev_data[inx][self.POSE_VAL] for e in [self.RIGHT_SHOULDER, self.RIGHT_HIP])
        curr_rightLine_corr_exist = all(e in pose_dix for e in [self.RIGHT_SHOULDER, self.RIGHT_HIP])

        left_angle = right_angle = 0

        if prev_leftLine_corr_exist and curr_leftLine_corr_exist:
            temp_left_vector = [[self._prev_data[inx][self.POSE_VAL][self.LEFT_SHOULDER],
                                self._prev_data[inx][self.POSE_VAL][self.LEFT_HIP]],
                                [pose_dix[self.LEFT_SHOULDER], pose_dix[self.LEFT_HIP]]]
            left_angle = self.calculate_angle(temp_left_vector)
            log.debug("Left shoulder-hip angle: %r", left_angle)

        if prev_rightLine_corr_exist and curr_rightLine_corr_exist:
            temp_right_vector = [[self._prev_data[inx][self.POSE_VAL][self.RIGHT_SHOULDER],
                                 self._prev_data[inx][self.POSE_VAL][self.RIGHT_HIP]],
                                 [pose_dix[self.RIGHT_SHOULDER], pose_dix[self.RIGHT_HIP]]]
            right_angle = self.calculate_angle(temp_right_vector)
            log.debug("Right shoulder-hip angle: %r", right_angle)

        angle_change = max(left_angle, right_angle)
        return angle_change

    def assign_prev_records(self, pose_dix, left_angle_with_yaxis,
                            rigth_angle_with_yaxis, now, thumbnail,
                            current_body_vector_score):

        curr_data = {self.POSE_VAL: pose_dix,
                     self.TIMESTAMP: now,
                     self.THUMBNAIL: thumbnail,
                     self.LEFT_ANGLE_WITH_YAXIS: left_angle_with_yaxis,
                     self.RIGHT_ANGLE_WITH_YAXIS: rigth_angle_with_yaxis,
                     self.BODY_VECTOR_SCORE: current_body_vector_score}

        self._prev_data[-2] = self._prev_data[-1]
        self._prev_data[-1] = curr_data

    def draw_lines(self, thumbnail, pose_dix, score):
        """Draw body lines if available. Return number of lines drawn."""
        # save an image with drawn lines for debugging
        draw = ImageDraw.Draw(thumbnail)
        path = None
        body_lines_drawn = 0

        if pose_dix is None:
            return body_lines_drawn

        if pose_dix.keys() >= {self.LEFT_SHOULDER, self.LEFT_HIP}:
            body_line = [tuple(pose_dix[self.LEFT_SHOULDER]),
                         tuple(pose_dix[self.LEFT_HIP])]
            draw.line(body_line, fill='red')
            body_lines_drawn += 1

        if pose_dix.keys() >= {self.RIGHT_SHOULDER, self.RIGHT_HIP}:
            body_line = [tuple(pose_dix[self.RIGHT_SHOULDER]),
                         tuple(pose_dix[self.RIGHT_HIP])]
            draw.line(body_line, fill='red')
            body_lines_drawn += 1

        # save a thumbnail for debugging
        timestr = int(time.monotonic()*1000)
        debug_image_file_name = \
            f'tmp-fall-detect-thumbnail-{timestr}-score-{score}.jpg'
        thumbnail.save(
                       Path(self._sys_data_dir, debug_image_file_name),
                       format='JPEG')
        return body_lines_drawn

    def get_line_angles_with_yaxis(self, pose_dix):
        '''
            Find the angle b/w shoulder-hip line with yaxis.
        '''
        y_axis_corr = [[0, 0], [0, self._pose_engine._tensor_image_height]]

        leftLine_corr_exist = all(e in pose_dix for e in [self.LEFT_SHOULDER, self.LEFT_HIP])
        rightLine_corr_exist = all(e in pose_dix for e in [self.RIGHT_SHOULDER, self.RIGHT_HIP])

        l_angle = r_angle = 0

        if leftLine_corr_exist:
            l_angle = self.calculate_angle([y_axis_corr,
                                           [pose_dix[self.LEFT_SHOULDER],
                                            pose_dix[self.LEFT_HIP]]])

        if rightLine_corr_exist:
            r_angle = self.calculate_angle([y_axis_corr,
                                           [pose_dix[self.RIGHT_SHOULDER],
                                            pose_dix[self.RIGHT_HIP]]])

        return (l_angle, r_angle)

    def estimate_spinal_vector_score(self, pose):
        pose_dix = {}
        is_leftVector = is_rightVector = False

        # Calculate leftVectorScore & rightVectorScore
        leftVectorScore = min(pose.keypoints[self.LEFT_SHOULDER].score,
                              pose.keypoints[self.LEFT_HIP].score)
        rightVectorScore = min(pose.keypoints[self.RIGHT_SHOULDER].score,
                               pose.keypoints[self.RIGHT_HIP].score)

        if leftVectorScore > self.confidence_threshold:
            is_leftVector = True
            pose_dix[self.LEFT_SHOULDER] = \
                pose.keypoints[self.LEFT_SHOULDER].yx
            pose_dix[self.LEFT_HIP] = pose.keypoints[self.LEFT_HIP].yx

        if rightVectorScore > self.confidence_threshold:
            is_rightVector = True
            pose_dix[self.RIGHT_SHOULDER] = \
                pose.keypoints[self.RIGHT_SHOULDER].yx
            pose_dix[self.RIGHT_HIP] = pose.keypoints[self.RIGHT_HIP].yx

        def find_spinalLine():
            left_spinal_x1 = (pose_dix[self.LEFT_SHOULDER][0] +
                              pose_dix[self.RIGHT_SHOULDER][0]) / 2
            left_spinal_y1 = (pose_dix[self.LEFT_SHOULDER][1] +
                              pose_dix[self.RIGHT_SHOULDER][1]) / 2

            right_spinal_x1 = (pose_dix[self.LEFT_HIP][0] +
                               pose_dix[self.RIGHT_HIP][0]) / 2
            right_spinal_y1 = (pose_dix[self.LEFT_HIP][1] +
                               pose_dix[self.RIGHT_HIP][1]) / 2

            return (left_spinal_x1, left_spinal_y1), \
                   (right_spinal_x1, right_spinal_y1)

        if is_leftVector and is_rightVector:
            spinalVectorEstimate = find_spinalLine()
            spinalVectorScore = (leftVectorScore + rightVectorScore) / 2.0
        elif is_leftVector:
            spinalVectorEstimate = pose_dix[self.LEFT_SHOULDER], \
                                   pose_dix[self.LEFT_HIP]
            # 10% score penalty in conficence as only \
            # left shoulder-hip line is detected
            spinalVectorScore = leftVectorScore * 0.9
        elif is_rightVector:
            spinalVectorEstimate = pose_dix[self.RIGHT_SHOULDER], \
                                   pose_dix[self.RIGHT_HIP]
            # 10% score penalty in conficence as only \
            # right shoulder-hip line is detected
            spinalVectorScore = rightVectorScore * 0.9
        else:
            spinalVectorScore = 0

        log.debug(f"Estimated spinal vector score: {spinalVectorScore}")
        return spinalVectorScore, pose_dix

    def fall_detect(self, image=None):
        assert image
        log.debug("Calling TF engine for inference")
        start_time = time.monotonic()

        now = time.monotonic()
        lapse = now - self._prev_data[-1][self.TIMESTAMP]

        if self._prev_data[-1][self.POSE_VAL] \
           and lapse < self.min_time_between_frames:
            log.debug("Received an image frame too soon after the previous \
                frame. Only %.2f ms apart.\
                Minimum %.2f ms distance required for fall detection.",
                lapse, self.min_time_between_frames)
            inference_result = None
            thumbnail = self._prev_data[-1][self.THUMBNAIL]
        else:
            # Detection using tensorflow posenet module
            pose, thumbnail, spinal_vector_score, pose_dix = \
                        self.find_keypoints(image)

            inference_result = None
            if not pose:
                log.debug(f"No pose detected or detection score does not meet \
                    confidence threshold of {self.confidence_threshold}.")
            else:
                inference_result = []

                current_body_vector_score = spinal_vector_score

                # Find line angle with vertcal axis
                left_angle_with_yaxis, rigth_angle_with_yaxis = \
                    self.get_line_angles_with_yaxis(pose_dix)

                # save an image with drawn lines for debugging
                if log.getEffectiveLevel() <= logging.DEBUG:
                    # development mode
                    self.draw_lines(thumbnail, pose_dix, spinal_vector_score)

                for t in [-1, -2]:
                    lapse = now - self._prev_data[t][self.TIMESTAMP]

                    if not self._prev_data[t][self.POSE_VAL] or \
                       lapse > self.max_time_between_frames:
                        log.debug("No recent pose to compare to. Will save \
                            this frame pose for subsequent comparison.")
                    elif not self.is_body_line_motion_downward(
                                                        left_angle_with_yaxis,
                                                        rigth_angle_with_yaxis,
                                                        inx=t):
                        log.debug("The body-line angle with vertical axis is \
                                    decreasing from the previous frame. \
                                    Not likely to be a fall.")
                    else:
                        leaning_angle = self.find_changes_in_angle(pose_dix,
                                                                   inx=t)

                        # Get leaning_probability by comparing leaning_angle
                        # with fall_factor probability.
                        leaning_probability = 1 \
                            if leaning_angle > self._fall_factor else 0

                        # Calculate fall score using average of current and \
                        # previous frame's body vector score with \
                        # leaning_probability
                        fall_score = leaning_probability * \
                            (self._prev_data[t][self.BODY_VECTOR_SCORE] +
                             current_body_vector_score) / 2

                        if fall_score >= self.confidence_threshold:
                            inference_result.append(('FALL', fall_score,
                                                     leaning_angle, pose_dix))
                            log.info("Fall detected: %r", inference_result)
                            break
                        else:
                            log.debug(f"No fall detected due to low \
                            confidence score:  \
                            {fall_score} < {self.confidence_threshold} \
                            min threshold.Inference result: {inference_result}")

                log.debug("Saving pose for subsequent comparison.")
                self.assign_prev_records(pose_dix, left_angle_with_yaxis,
                                         rigth_angle_with_yaxis, now,
                                         thumbnail,
                                         current_body_vector_score)

                # log.debug("Logging stats")

        self.log_stats(start_time=start_time)
        log.debug("thumbnail: %r", thumbnail)
        return inference_result, thumbnail

    def convert_inference_result(self, inference_result):
        inf_json = []

        if inference_result:
            for inf in inference_result:
                label, confidence, leaning_angle, keypoint_corr = inf
                log.info('label: %s , confidence: %.0f, leaning_angle: %.0f, \
                         keypoint_corr: %s',
                         label,
                         confidence,
                         leaning_angle,
                         keypoint_corr)
                one_inf = {
                    'label': label,
                    'confidence': confidence,
                    'leaning_angle': leaning_angle,
                    'keypoint_corr': {
                        'left shoulder': keypoint_corr.get('left shoulder',
                                                           None),
                        'left hip': keypoint_corr.get('left hip', None),
                        'right shoulder': keypoint_corr.get('right shoulder',
                                                            None),
                        'right hip': keypoint_corr.get('right hip', None)
                    }
                }
                inf_json.append(one_inf)

        return inf_json
