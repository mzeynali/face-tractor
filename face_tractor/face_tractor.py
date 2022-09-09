import os
from pathlib import Path
from typing import List

import mediapipe as mp
import numpy as np

from .head_pose import HeadPoseNormalizer
from .face_cam import Camera, Face, FaceModelMediaPipe, detect_faces_mediapipe

file_dir = os.path.dirname(__file__)


class FaceTractor:
    def __init__(
        self,
        normalized_distance: float = 0.6,
        static_image_mode: bool = True,
        refine_landmarks: bool = True,
        camera_params: str = "sample_params.yaml",
        normalized_camera_params: str = "eth-xgaze.yaml",
    ):
        """
        Args:
            normalized_distance: distance between the center of the face
                and virtual camera. by zooming in/out. image size (default
                :224x224) cames from virtual camera parameters `camera_params`.
            static_image_mode: if True, faceMesh tracker will be disabled.
            camera_params: path to camera parameters file.
            normalized_camera_params: path to camera parameters file for
                normalized face.
        """
        # necessary directories
        base_dir = Path(file_dir)
        camera_params = base_dir / camera_params
        normalized_camera_params = base_dir / normalized_camera_params

        # initialize detector, face mode and head pose normalizer
        self.detector = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            static_image_mode=static_image_mode,
            refine_landmarks=refine_landmarks,
        )
        self.face_model_3d = FaceModelMediaPipe()
        self.head_pose_normalizer = HeadPoseNormalizer(
            Camera(camera_params.as_posix()),
            Camera(normalized_camera_params.as_posix()),
            normalized_distance=normalized_distance,
        )

    def normalized_faces(self, frame: np.ndarray) -> List[Face]:
        """
        Args:
            frame: input frame.
        Returns:
            List of faces with `normalized_image: np.ndarray` property.
        """
        faces = detect_faces_mediapipe(self.detector, frame)
        for face in faces:
            self.face_model_3d.estimate_head_pose(
                face, self.head_pose_normalizer.camera
            )
            self.face_model_3d.compute_3d_pose(face)
            self.face_model_3d.compute_face_eye_centers(face, "ETH-XGaze")
            self.head_pose_normalizer.normalize(frame, face)
        return faces
