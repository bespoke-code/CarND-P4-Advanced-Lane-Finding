import numpy as np


class Calibration:
    def __init__(self):
        """
        Initializes calibration parameters:
         - the intrinsic matrix K(3x3) is initialized as an Identity matrix
         - all distortion coefficients are initialized to zero
         - width and height are set to zero
        """
        self.fx = 1.
        self.fy = 1.
        self.cx = 0.
        self.cy = 0.

        self.k1 = 0.
        self.k2 = 0.
        self.k3 = 0.
        self.p1 = 0.
        self.p2 = 0.

        self.width = 0.
        self.height = 0.

    def calibrationMatrix(self):
        """
        Returns the intrinsic parameters matrix K(3x3), containing fx, fy, cx and cy.
        :return: The intrinsic matrix K as a 3x3 numpy array.
        """
        return np.array([[self.fx,      0., self.cx],
                         [     0., self.fy, self.cy],
                         [     0.,      0.,      1.]], dtype=np.double)

    def distortionCoefficients(self):
        """
        Returns the distortion coefficients array D(1x5), containing k1, k2, p1, p2, k3
        :return: The distortion coefficients D as a 1x5 numpy array.
        """
        return np.array([[self.k1, self.k2, self.p1, self.p2, self.k3]], dtype=np.double)

    def imageSize(self):
        """
        Returns the valid image size for which the current calibration settings are valid.
        :return: The image size as a tuple (width, height)
        """
        return [self.width, self.height]

    def isSizeValid(self, w, h):
        """
        Checks if the current calibration is valid for an image of a given size.
        :param w: An image's width
        :param h: An image's height
        :return: True if the given image size can be correctly calibrated using this calibration parameters set.
        """
        return (w == self.width) and (h == self.height)


class CalibrationFactory:
    def __init__(self):
        self.calibration = Calibration()

    def setImageSize(self, w, h):
        self.calibration.width = w
        self.calibration.height = h
        return self

    def setDistortionCoeffs(self, k1, k2, p1, p2, k3=0.):
        self.calibration.k1 = k1
        self.calibration.k2 = k2
        self.calibration.k3 = k3
        self.calibration.p1 = p1
        self.calibration.p2 = p2
        return self

    def setCameraMatrixParams(self, fx, fy, cx, cy):
        self.calibration.fx = fx
        self.calibration.fy = fy
        self.calibration.cx = cx
        self.calibration.cy = cy
        return self

    def build(self):
        return self.calibration

    @staticmethod
    def SamsungGalaxyS7Calibration1440p():
        return CalibrationFactory().setImageSize(w=2560,
                                                 h=1440)\
                                    .setCameraMatrixParams(fx=1.98217212e+03,  # TODO: Check parameters!
                                                           fy=1.98591413e+03,
                                                           cx=1.27265921e+03,
                                                           cy=7.39646965e+02)\
                                    .setDistortionCoeffs(k1=0.33640685,
                                                         k2=-0.9461401,
                                                         p1=0.00152668,
                                                         p2=0.00282108,
                                                         k3=0.66146681)\
                                    .build()

    @staticmethod
    def SamsungGalaxyS7Calibration720p():
        return CalibrationFactory().setImageSize(w=1280,
                                                 h=720)\
                                    .setCameraMatrixParams(fx=1.98217212e+03,  # TODO: Calibrate at 720p!
                                                           fy=1.98591413e+03,
                                                           cx=1.27265921e+03,
                                                           cy=7.39646965e+02)\
                                    .setDistortionCoeffs(k1=0.33640685,
                                                         k2=-0.9461401,
                                                         p1=0.00152668,
                                                         p2=0.00282108,
                                                         k3=0.66146681)\
                                    .build()


