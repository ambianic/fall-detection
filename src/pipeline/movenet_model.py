from src.pipeline.pose_base import AbstractPoseModel
import numpy as np


class Movenet(AbstractPoseModel):

    def __init__(self, tfengine):
        super().__init__(tfengine)


    def parse_output(self, keypoints_with_scores, height, width):
    
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

        keypoints_with_scores = self.tf_interpreter().get_tensor(self._tfengine.output_details[0]['index'])
        kps = self.parse_output(keypoints_with_scores, self._tensor_image_height, self._tensor_image_width)

        return kps, template_image, thumbnail