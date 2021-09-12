from src.pipeline.pose_base import AbstractPoseModel
import numpy as np
import time

class Posenet_MobileNet(AbstractPoseModel):
    '''The class for pose estimation using Posenet Mobilenet implementation.'''

    def __init__(self, tfengine):
        super().__init__(tfengine)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def parse_output(self, heatmap_data, offset_data):
        '''
            Parse Output of TFLite model and get keypoints with score.
        '''

        joint_num = heatmap_data.shape[-1]
        pose_kps = np.zeros((joint_num, 3), np.float32)

        for i in range(heatmap_data.shape[-1]):

            joint_heatmap = heatmap_data[..., i]
            max_val_pos = np.squeeze(
                np.argwhere(joint_heatmap == np.max(joint_heatmap)))
            remap_pos = np.array(max_val_pos/8*self._tensor_image_height,
                                 dtype=np.int32)
            pose_kps[i, 0] = int(remap_pos[0] + offset_data[max_val_pos[0],
                                 max_val_pos[1], i])
            pose_kps[i, 1] = int(remap_pos[1] + offset_data[max_val_pos[0],
                                 max_val_pos[1], i+joint_num])
            max_prob = np.max(joint_heatmap)
            
            pose_kps[i,2] = self.sigmoid(max_prob)
            

        return pose_kps


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
            template_input = (np.float32(template_input) - 127.5) / 127.5

        self.tf_interpreter().\
            set_tensor(self._tfengine.input_details[0]['index'],
                       template_input)
        self.tf_interpreter().invoke()

        template_output_data = self.tf_interpreter().\
            get_tensor(self._tfengine.output_details[0]['index'])
        template_offset_data = self.tf_interpreter().\
            get_tensor(self._tfengine.output_details[1]['index'])

        template_heatmaps = np.squeeze(template_output_data)
        template_offsets = np.squeeze(template_offset_data)

        kps = self.parse_output(template_heatmaps, template_offsets)

        _inference_time = time.process_time() - start_time

        return kps, template_image, thumbnail, _inference_time