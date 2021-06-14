import os
import time

from src.pipeline.fall_detect import FallDetector


def _fall_detect_config():

    _dir = os.path.dirname(os.path.abspath(__file__))
    _good_tflite_model = os.path.join(
        _dir,
        'ai_models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
        )
    _good_edgetpu_model = os.path.join(
        _dir,
        'ai_models/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite'
        )
    _good_labels = 'ai_models/pose_labels.txt'
    config = {
        'model': {
            'tflite': _good_tflite_model,
            'edgetpu': _good_edgetpu_model,
            },
        'labels': _good_labels,
        'top_k': 3,
        'confidence_threshold': 0.6,
    }
    return config


def Fall_prediction(img_1,img_2,img_3=None):
    
    config = _fall_detect_config()
    result = None
    
    fall_detector = FallDetector(**config)

    def process_response(response):
        nonlocal result
        for res in response:
            result = res['inference_result']

    process_response(fall_detector.process_sample(image=img_1))
        
    time.sleep(fall_detector.min_time_between_frames)
    
    process_response(fall_detector.process_sample(image=img_2))
    
    if len(result) == 1:
        category = result[0]['label']
        confidence = result[0]['confidence']
        angle = result[0]['leaning_angle']
        keypoint_corr = result[0]['keypoint_corr']

        dict_res = {}
        dict_res["category"] = category
        dict_res["confidence"] = confidence
        dict_res["angle"] = angle
        dict_res["keypoint_corr"] = keypoint_corr
        return dict_res

    else:

        if img_3:
            
            time.sleep(fall_detector.min_time_between_frames)
            process_response(fall_detector.process_sample(image=img_3))

            if len(result) == 1:

                category = result[0]['label']
                confidence = result[0]['confidence']
                angle = result[0]['leaning_angle']
                keypoint_corr = result[0]['keypoint_corr']
                
                dict_res = {}
                dict_res["category"] = category
                dict_res["confidence"] = confidence
                dict_res["angle"] = angle
                dict_res["keypoint_corr"] = keypoint_corr
                return dict_res
        
    return None
