# Interactive co-manipulation agent

<img src="./images/nao.jpg" width="50%">

A handover algorithm based on the detection of colored objects in the scene
and on deciding about the robot’s ability to execute stable grasping.
Takes into account the relative positioning and orientation of the object inside the field of 
view of the robot's on-board camera.

### Robotic platform

<a href="https://www.ald.softbankrobotics.com/en/robots/nao" target="_blank">NAO</a>
Humanoid robot [NAO](https://www.ald.softbankrobotics.com/en/robots/nao){:target="_blank"}  
* vision: monocular RGB camera
* manipulator: 5-DOF arms, 1-DOF hands

### Detection
The object detection is based on simple color-based segmentation:

![detection_pipeline](./images/detection_pipeline.png)

* RGB → HSV
* thresholding
* morphological filtering:
<a href="https://www.codecogs.com/eqnedit.php?latex=I&space;\leftarrow&space;(I&space;\circ&space;B)&space;\bullet&space;B" target="_blank"><img src="https://latex.codecogs.com/svg.latex?I&space;\leftarrow&space;(I&space;\circ&space;B)&space;\bullet&space;B" title="I \leftarrow (I \circ B) \bullet B" /></a>
* select max-area contour

### Classification
The robot extracts visual geometric features of the object (centroid, width, height, orientation, etc)
and based on training examples of good, average and bad handovers decides if it can grasp it.
The classifier used in a Support Vector Machine with gaussian radial basis kernel.

<img src="./images/conf_3c_b.png" width="40%">

### Visual servoing and end-effector adaptation
The robot's head follows the object by adjusting its yaw and pitch angles through
a simple proportional control. Given that the training samples were captures in a fixed head orientation
the hand's pose must adapt to compensate the camera frame-hand frame change.

<img src="./images/features_move.png" width="80%">

# Dependencies

* numpy
* scikit-learn
* opencv
* naoqi