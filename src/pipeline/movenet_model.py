from src.pipeline.pose_base import AbstractPoseModel
import numpy as np
import time


class Movenet(AbstractPoseModel):
    '''The class for pose estimation using Movenet implementation.'''

    def __init__(self, tfengine):
        super().__init__(tfengine)


    def parse_output(self, keypoints_with_scores, height, width):
        '''
            Parse Output of TFLite model and get keypoints with score.
        '''
    
        keypoints_all = []
        num_instances, _, _, _ = keypoints_with_scores.shape
        
        for idx in range(num_instances):
            
            kpts_y = keypoints_with_scores[0, idx, :, 1]
            kpts_x = keypoints_with_scores[0, idx, :, 0]
            
            kpts_scores = keypoints_with_scores[0, idx, :, 2]
            
            kpts_absolute_xy = np.stack([width * np.array(kpts_x), height * np.array(kpts_y), kpts_scores], axis=-1)
            keypoints_all.append(kpts_absolute_xy)
            
        if keypoints_all:
            keypoints_xy = np.concatenate(keypoints_all, axis=0)
        else:
            keypoints_xy = np.zeros((0, 17, 2))

        return keypoints_xy


    def execute_model(self, img):
        ''' Run TFLite model.
        
        :Parameters:
        ----------
        img: PIL.Image
            Input Image for AI model detection.
        :Returns:
        -------
        kps:
            A list of Pose objects with keypoints and confidence scores
        template_image: PIL.Image
            Input resized image.
        thumbnail: PIL.Image
            Thumbnail input image
        _inference_time: float
            Model inference time in seconds
        '''
        
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

        start_time = time.process_time()

        template_input = np.expand_dims(template_image.copy(), axis=0)
        floating_model = self._tfengine.input_details[0]['dtype'] == np.float32

        if floating_model:
            template_input = template_input.astype(np.float32)
            
        self.tf_interpreter().\
            set_tensor(self._tfengine.input_details[0]['index'],
                       template_input)
        self.tf_interpreter().invoke()

        keypoints_with_scores = self.tf_interpreter().get_tensor(self._tfengine.output_details[0]['index'])
        kps = self.parse_output(keypoints_with_scores, self._tensor_image_height, self._tensor_image_width)


        _inference_time = time.process_time() - start_time

        return kps, template_image, thumbnail, _inference_time