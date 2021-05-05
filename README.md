# People Fall Detection

_This repo is the new home of the fall detection model used in Ambianic Edge. We are preparing to move the code over from the [ambianic-edge repo](https://github.com/ambianic/ambianic-edge/blob/master/src/ambianic/pipeline/ai/fall_detect.py) and make it available as a standalone library._

# TODO

- [x] Standalone Python ML library for people fall detection based on Tensorflow and [PoseNet 2.0](https://github.com/tensorflow/tfjs-models/tree/master/posenet).
- [ ] Python source code and wheel package published on PyPi 
- [x] Jupyter Notebook to interactively test and experiement with the model
- [ ] CI & test suite
- [ ] Training and testing data sets
- [ ] Third party ML models used as building blocks

# Project motivation

For many adults, one of the most difficult decisions to make is how to care for an elderly parent or relative that needs assistance. The AARP has found that almost 90% of folks over the age of 65 prefer to remain independent by living out their golden years at home. 

Whether living alone or with family members, elderly parents need constant monitoring. Why? This is because as they age, their risk to potentially life-threatening accidents increases. 

In fact, a slew of researches reveal that seniors are more prone to fall than other age classes. Falls are the leading cause of fatal injury and the most common cause of nonfatal trauma-related hospital admissions among older adults.

In a recent [guest blog post for Linux Foundation AI & Data](https://lfaidata.foundation/blog/2021/01/14/people-fall-detection-via-privacy-preserving-ai/) we shared the background of the problem and current market solutions.

# How it works

The Fall Detection algorithm fits well with the Ambianic framework of privacy preserving AI for home monitoring and automation. The following diagram illustrates the overall system architecture. 
End users install an Ambianic Box to constantly monitor a fall risk area of the house. If a fall is detected, family and caregivers are instantly notified that their loved one has fallen down.

![Fall Detection high level system architecture](https://user-images.githubusercontent.com/2234901/112542950-25d6d300-8d83-11eb-9048-feabd64de22d.png)

In the current design we use a combination of the [PoseNet 2.0](https://github.com/tensorflow/tfjs-models/tree/master/posenet) Deep Neural Network model and domain specific heuristics to estimate a fall occurance. The following diagram illustates the main steps.

[![Fall Detection AI flow](https://user-images.githubusercontent.com/2234901/112545190-ea89d380-8d85-11eb-8e2c-7a6b104d159e.png)](https://drive.google.com/file/d/1sr2OcEWsGzoxJb4PwCIXOuEo7a5ubAxG/view?usp=sharing)

## Experiment

Experiment with the fall-detection module using simple script, jupyter-notebook or command line input(CLI) by feeding 2 or 3 images. The input images should be spaced about 1 second apart.

###### Run a Python Script

```
python3 demo-fall-detection.py
```

###### Run a `Demo.ipynb` jupyter-notebook

###### Exceute below command for CLI usage

To test fall-detection using the CLI for 2 images:

```
python3 demo-fall-detection-cmd.py --image_1 Images/fall_img_1.png --image_2 Images/fall_img_2.png
```

To test fall-detection using the CLI for 3 images:

```
python3 demo-fall-detection-cmd.py --image_1 Images/fall_img_1.png --image_2 Images/fall_img_2.png --image_3 Images/fall_img_3.png
```

# Limitations

Based on testing and user feedback we are aware of the following limitations for the Fall Detection algorithm:

- Distance from monitored area: Optimal about 15-25 feet (5-8 meters). The camera has to be able to see the whole person in standing position before the fall and fallen position after the fall. If the camera is not able to see a substantial part of the body, the Fall Detection model may not have sufficient confidence in its estimation of the situation.
- Camera hight: Optimal at about human eye level: about 4-7ft (1-2 meters). If the camera is angled too low or too high overhead , the PoseNet model is not able to estimate body keypoints with confidence.
- Lighting condition: Good lighting is required for optimal performance. Dim light reduces PoseNet's confidence in keypoint estimates.
- Single person: The model is optimized for situation when a person is home alone. If there are multiple people in the camera view, that may confuse the model and lead to false fall detections.
- No clutter: The model performs best when the area being monitored is relatively clear of various small objects. Areas cluttered with objects may confuse the model that some of these objects look like people.

# Future work

The current version of the Fall Detector uses PoseNet Mobilnetv1. There are newer, more powerful models such as PoseNet 2.0 Resnet50 and BlazePose that are able to estimate body keypoints in a broader variety of situations. However they can be also more CPU resource intensive, which reduces the FPS we can process. We are working on testing some of these new models and converting them to downsized IoT optimized versions, with minimum loss of accuracy and speed tradeoff. You can track our work [here](https://github.com/ambianic/fall-detection/issues/5).

As we work with families and caregivers to test the system in real world scenarious, we expect to develop better intuition for the key factors that determine a fall in a sequence of video frames. 
Eventually we expect to replace some of the current heuristics with learned models that are able to more precisely distinguish between true falls and non-falls (e.g. bending over or squating to tie shoes).

Ideas and constructive criticism are welcome. Feel free to join the discussion on [Slack](https://ambianicai.slack.com/join/shared_invite/zt-eosk4tv5-~GR3Sm7ccGbv1R7IEpk7OQ#/), open a [github issue](https://github.com/ambianic/fall-detection/issues) or PR draft.
