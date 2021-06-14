import sys
import os
sys.path.append(os.path.abspath('.'))

from PIL import Image
from fall_prediction import Fall_prediction

def _get_image(file_name=None):
    assert file_name

    img = Image.open(file_name)
    return img


def test_fall_detetcion_with_two_images_case_1():
    "Expected to not detect a fall with two consecutive frames"

    img1 = _get_image(file_name='Images/fall_img_1.png')
    img2 = _get_image(file_name='Images/fall_img_2.png')

    result = Fall_prediction(img1, img2)
    
    assert result is None

def test_fall_detetcion_with_two_images_case_2():
    "Expected to detect a fall with two consecutive frames"

    img1 = _get_image(file_name='Images/fall_img_1.png')
    img2 = _get_image(file_name='Images/fall_img_3.png')

    result = Fall_prediction(img1, img2)
    
    assert result

    confidence = result['confidence']
    angle = result['angle']
    keypoint_corr = result['keypoint_corr']

    assert keypoint_corr
    assert confidence > 0.3
    assert angle > 60


def test_fall_detetcion_with_three_images_case_1():
    """
    Expected to detect a fall from the first and third frame by given three consecutive frames.
    """

    img1 = _get_image(file_name='Images/fall_img_1.png')
    img2 = _get_image(file_name='Images/fall_img_2.png')
    img3 = _get_image(file_name='Images/fall_img_3.png')

    result = Fall_prediction(img1, img2, img3)
    
    assert result

    confidence = result['confidence']
    angle = result['angle']
    keypoint_corr = result['keypoint_corr']

    assert keypoint_corr
    assert confidence > 0.3
    assert angle > 60

def test_fall_detetcion_with_three_images_case_2():
    """
    Expected to detect a fall from the first and second frame by given three consecutive frames.
    """
    img1 = _get_image(file_name='Images/fall_img_1.png')
    img2 = _get_image(file_name='Images/fall_img_3.png')
    img3 = _get_image(file_name='Images/fall_img_4.png')

    result = Fall_prediction(img1, img2, img3)
    
    assert result

    confidence = result['confidence']
    angle = result['angle']
    keypoint_corr = result['keypoint_corr']

    assert keypoint_corr
    assert confidence > 0.3
    assert angle > 60


def test_fall_detetcion_with_three_images_case_3():
    """
    Expected to detect a fall from the second and third frame by given three consecutive frames.
    """

    img1 = _get_image(file_name='Images/background.jpg')
    img2 = _get_image(file_name='Images/fall_img_8.png')
    img3 = _get_image(file_name='Images/fall_img_9.png')

    result = Fall_prediction(img1, img2, img3)
    
    assert result

    confidence = result['confidence']
    angle = result['angle']
    keypoint_corr = result['keypoint_corr']

    assert keypoint_corr
    assert confidence > 0.3
    assert angle > 60

def test_fall_detetcion_with_three_images_case_4():
    "Expected to not detect a fall with three consecutive frames"

    img1 = _get_image(file_name='Images/background.jpg')
    img2 = _get_image(file_name='Images/fall_img_5.png')
    img3 = _get_image(file_name='Images/fall_img_6.png')

    result = Fall_prediction(img1, img2, img3)
    
    assert result is None

    