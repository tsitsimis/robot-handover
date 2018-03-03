# Interactive co-manipulation agent

A handover algorithm based on the detection of colored objects in the scene
and on deciding about the robot’s ability to execute stable grasping.
Takes into account the relative positioning and orientation of the object inside the field of 
view of the robot's on-board camera.

### Robotic platform

Humanoid robot [NAO](https://www.ald.softbankrobotics.com/en/robots/nao)  
* vision: monocular camera
* manipulator: 5-DOF arms, 1-DOF hands

# Dependencies

* numpy
* scikit-learn
* opencv
* naoqi