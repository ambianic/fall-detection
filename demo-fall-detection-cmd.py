import argparse
from pathlib import Path
from PIL import Image
from fall_prediction import Fall_prediction
import numpy as np
import json
import time
import yaml

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super(JsonEncoder, self).default(obj)

parser = argparse.ArgumentParser()
parser.add_argument("--image_1", type=Path, help='Path to the First Image', required=True)
parser.add_argument("--image_2", type=Path, help='Path to the Second Image', required=True)
parser.add_argument("--image_3", type=Path, help='Path to the Third Image')

p = parser.parse_args()
img1 = Image.open(p.image_1)
img2 = Image.open(p.image_2)
img3 = Image.open(p.image_3) if p.image_3 else None

response = Fall_prediction(img1, img2, img3)

if response:

    print("There is", response['category'])
    print("Confidence :", response['confidence'])
    print("Angle : ", response['angle'])
    print("Keypoint_corr :", response['keypoint_corr'])

    time_str = time.strftime("%Y%m%d-%H%M%S")
    json_str = json.dumps(response, cls=JsonEncoder)

    with open(f"tmp/{time_str}.yaml","w",) as file:
        yaml.dump(json.loads(json_str), file)
else:
     print("There is no fall detetcion...")

