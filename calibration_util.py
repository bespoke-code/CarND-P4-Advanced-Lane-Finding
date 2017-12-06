import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt


def calibrateCamera(image_shape, objpoints, imgpoints):
    ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape[::-1], None, None)
    return camera_matrix, dist


def undistort_image(image, camera_matrix, dist):
    return cv2.undistort(image, cameraMatrix=camera_matrix, distCoeffs=dist, newCameraMatrix=camera_matrix)


if __name__ == '__main__':
    # Full calibration pipeline is here

    # Get all calibration images
    calib_folder = './camera_cal/'
    calib_files = glob.glob(calib_folder + 'calibration*.jpg')
    calib_files_shape = cv2.imread(calib_files[0]).shape

    # Define checkerboard pattern
    pattern_size = (9, 6)

    # Define checkerboard pattern points
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Collect object points and image points from images
    object_points = []
    img_points = []

    for image_path in calib_files:
        image = plt.imread(image_path)
        found, corners = cv2.findChessboardCorners(image, patternSize=pattern_size, corners=None)
        if found:
            objp.append(objp)
            img_points.append(corners)
        else:
            print('Can\'t find chessboard pattern in {calib_img}!'.format(calib_img=image))
    print('Points collected! Running camera calibration...')

    camera_matrix, dst_coeffs = calibrateCamera(calib_files_shape, object_points, img_points)
    print('Camera calibration completed.')
    print('Camera matrix:', camera_matrix)
    print('Distortion coefficients', dst_coeffs)

