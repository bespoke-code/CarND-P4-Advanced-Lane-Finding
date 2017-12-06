import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt


def calibrateCamera(image_shape, objpoints, imgpoints):
    ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape[1::-1], None, None)
    return camera_matrix, dist


def undistort_image(image, camera_matrix, dist):
    return cv2.undistort(image, cameraMatrix=camera_matrix, distCoeffs=dist, newCameraMatrix=camera_matrix)


def getCalibParams(calib_files, pattern_size=(9,6)):

    assert (len(calib_files) != 0)

    calib_files_shape = cv2.imread(calib_files[0]).shape

    # Define checkerboard pattern points
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    object_points = []
    img_points = []

    # Iterate over all the images available and grab the image points
    for image_path in calib_files:
        image = plt.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(image, patternSize=pattern_size, corners=None)
        if found:
            object_points.append(objp)
            img_points.append(corners)
        else:
            print('Can\'t find chessboard pattern in {calib_img}!'.format(calib_img=image_path))
    print('Points collected! Running camera calibration...')

    camera_matrix, dst_coeffs = calibrateCamera(calib_files_shape, object_points, img_points)
    print('Camera calibration completed.')
    print('Camera matrix:', camera_matrix)
    print('Distortion coefficients', dst_coeffs)
    return camera_matrix, dst_coeffs


if __name__ == '__main__':
    # Full calibration pipeline is here

    # Get all calibration images
    calib_folder = './camera_cal/'
    calib_files = glob.glob(calib_folder + 'calibration*.jpg')


    # Define checkerboard pattern
    pattern_size = (9, 6)

    # Collect object points and image points from images
    camera_matrix, distort_coeffs = getCalibParams(calib_files, pattern_size)
