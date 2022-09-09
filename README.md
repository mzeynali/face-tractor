# Face Tractor

Face detection, facial landmarks, head pose, gaze estimation, and... .
This is base module for facial analysis. Features where used in this project will expand during the development based on the need.  Other projects which need facial analysis must be compatible with this module.


This work heavily inspired by [pytorch_mpiigaze_demo](https://github.com/hysts/pytorch_mpiigaze_demo) project.

## Installation
Clone and install the package:
```bash
git clone https://github.com/mzeynali/face-tractor.git
cd face_tractor
pip install -e .
```


## Usage

Just import and initialize the `FaceTractor` class and feed it with the `numpy.ndarray` objects as image.

```python
import cv2

from face_tractor import FaceTractor

"""
Change padding (in fact zooming in/out)
by changing normalized_distance parameter
"""
face_extractor = FaceTractor(normalized_distance=0.6)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    faces = face_extractor.normalized_faces(frame)
```

normalized_distance is a parameter which controls the zoom in/out of the face. It is a float value between 0 and 1. Actually the distance between the face and the virtual camera.