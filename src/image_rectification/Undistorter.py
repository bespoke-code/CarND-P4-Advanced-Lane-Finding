import numpy as np
from cv2 import undistort
from src.image_rectification.CameraCalibration import Calibration


class Undistorter:
    def __init__(self, calib: Calibration=None):
        if calib is None:
            self.camera_matrix = np.eye(3)
            self.distort_coeffs = np.zeros((1, 5))
            self.image_size = (0., 0.)
            self.isValid = False
        else:
            self.camera_matrix = calib.calibrationMatrix()
            self.distort_coeffs = calib.distortionCoefficients()
            self.image_size = calib.imageSize()
            self.isValid = True

    def isValid(self) -> bool:
        return self.isValid

    def undistort(self, image: np.array) -> np.array:
        assert(image.shape[0:2] == self.image_size)

        return undistort(image,
                         cameraMatrix=self.camera_matrix,
                         distCoeffs=self.distort_coeffs,
                         newCameraMatrix=self.camera_matrix
                         )
