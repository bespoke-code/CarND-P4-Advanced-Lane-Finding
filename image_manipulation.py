import cv2
import numpy as np
import calibration_util as calib


# Perspective transformations
#######################################################################
def perspective_warp(width=1280, height=720):
    '''
    Calculates warp and unwarp matrices based on a fixed set of points.
    :param width: Image width.
    :param height: Image height.
    :return: warp and unwarp matrices, each (3x3) numpy 2D arrays
    '''

    top_left =[594, 450] # [520, 500] # [7/16*width, 6/10*height]
    top_right = [687, 450]#[768, 500] # [9/16*width, 6/10*height]
    bottom_left = [262, 670] # [1/16*width, 9/10*height]
    bottom_right = [1044, 670] #[15/16*width, 9/10*height]
    src_points = np.float32([top_left, top_right, bottom_left, bottom_right])
    dst_points = np.float32([
        [4/16*width, 0], # top left
        [12/16*width, 0], # top right
        [4/16*width, height], # bottom left
        [12/16*width, height]  # bottom right
    ])

    warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    unwarp_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    return warp_matrix, unwarp_matrix


# Transform to a birds-eye perspective
def warp_image(image, warp_matrix):
    width = image.shape[1]
    height = image.shape[0]
    warped_image = cv2.warpPerspective(image, warp_matrix, (width, height))
    return warped_image


def getWarpMatrix():
    return [[-4.86204634e-01, -1.49507925e+00, 9.38982700e+02],
            [-5.88418203e-15, -1.94426603e+00, 8.74919714e+02],
            [-8.89045781e-18, -2.37922580e-03, 1.00000000e+00]]


def getUnwarpMatrix():
    return [[1.45312500e-01, -7.81724211e-01, 5.47500000e+02],
            [-3.55271368e-15, -5.14332907e-01, 4.50000000e+02],
            [-8.23993651e-18, -1.22371412e-03, 1.00000000e+00]]


# Image gradient and colour manipulation
#######################################################################
def sobel(gray, direction):
    if direction == 'x': # Sobel X
        return cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        return cv2.Sobel(gray, cv2.CV_64F, 0, 1)


# From the Udacity Advanced Lane Finding lectures
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sobel_binary


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    magnitude_sobel = np.sqrt((sobelx * sobelx + sobely * sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * magnitude_sobel / np.max(magnitude_sobel))
    # 5) Create a binary mask where mag thresholds are met
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sobel_binary


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction_grad = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    sobel_binary = np.zeros_like(direction_grad)
    sobel_binary[(direction_grad >= thresh[0]) & (direction_grad <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sobel_binary


def sobel_select(img, thresh=(20, 100)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_new = img[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(img_new, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    sobel_x_binary = np.zeros_like(scaled_sobel)
    sobel_x_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sobel_x_binary


# Color and gradient thresholds
#######################################################################
# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary


# Yellow lane lines extraction
def yellow_lines_RGB(image):
    color_select_img = np.copy(image)
    red_threshold = 220
    green_threshold = 180
    blue_threshold = 40

    # Identify pixels below the threshold. Black 'em out
    colour_thresholds = (image[:,:,0] < red_threshold) | \
                 (image[:,:,1] < green_threshold) | \
                 (image[:,:,2] < blue_threshold)
    color_select_img[colour_thresholds] = [0,0,0]
    color_select_img = cv2.cvtColor(color_select_img, cv2.COLOR_RGB2GRAY)
    slice = color_select_img[:,:] > 0
    color_select_img[slice] = 255
    return color_select_img


# White lane lines extraction
def white_lines_RGB(image):
    color_select_img = np.copy(image)
    red_threshold = 200
    green_threshold = 185
    blue_threshold = 199

    # Identify pixels below the threshold. Black 'em out
    colour_thresholds = (image[:,:,0] < red_threshold) | \
                 (image[:,:,1] < green_threshold) | \
                 (image[:,:,2] < blue_threshold)
    color_select_img[colour_thresholds] = [0,0,0]
    color_select_img = cv2.cvtColor(color_select_img, cv2.COLOR_RGB2GRAY)
    slice = color_select_img[:,:] > 0
    color_select_img[slice] = 255
    return color_select_img


def combined_sobel(img, ksize=3, thresh=(0,255), mag_thresh=(0,255), dir_thresh=(0, np.pi/2)):
    # Choose a larger k-size (odd number) to get smoother gradient measurements
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=thresh)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=thresh)
    mag_binary = mag_threshold(img, sobel_kernel=ksize, mag_thresh=mag_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=dir_thresh)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


# The final image processing pipeline
#######################################################################
def processFrame(frame):
    """
    Image processing pipeline for a video frame / standalone road image.
    :param frame: A video frame or an RGB camera image.
    :return: A mask showing the lane lines on the road image.
    """
    camera_matrix = calib.getCameraCalibration()
    dist_coeffs = calib.getDistortionCoeffs()
    warp_matrix = getWarpMatrix()

    image = cv2.undistort(frame, np.array(camera_matrix), np.array(dist_coeffs), None, np.array(camera_matrix))
    image = warp_image(image, warp_matrix)
    color_mask = cv2.add(yellow_lines_RGB(image), white_lines_RGB(image))
    color_mask[color_mask > 0] = 255

    sobel_mask = cv2.add(
        combined_sobel(
            cv2.blur(image, (9, 9)), ksize=3,
            thresh=(15, 90),
            mag_thresh=(10, 90),
            dir_thresh=(0.8, 1.)
        ),
        combined_sobel(
            cv2.blur(image, (9, 9)), ksize=3,
            thresh=(15, 90),
            mag_thresh=(10, 90),
            dir_thresh=(0.01, 0.2)
        )
    )
    sobel_mask[sobel_mask > 0] = 255

    kernel = np.ones((3, 3), np.uint8)
    sobel_mask = cv2.erode(sobel_mask, kernel, iterations=1)
    sobel_mask = cv2.morphologyEx(sobel_mask, cv2.MORPH_OPEN, kernel)

    hls_mask = hls_select(cv2.blur(image, (9, 9)), thresh=(115, 255))
    hls_mask[hls_mask > 0] = 255

    result_mask = np.zeros_like(hls_mask)
    result_mask[sobel_mask > 0] = 255
    result_mask[hls_mask > 0] = 255
    result_mask[color_mask > 0] = 255
    result_mask[:, result_mask.shape[1] - 100:result_mask.shape[1]] = 0
    result_mask[:, :200] = 0
    result_mask[500:result_mask.shape[0], 450:900] = 0
    return result_mask


def frameOverlay(frame, left_line, right_line, width=1280, height=720, color=(0, 255, 0)):
    # Create an image to draw the lines on
    warp_zero = np.zeros((width, height), dtype=np.uint8)
    #warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.fitx, left_line.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.fitx, right_line.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), color)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, getUnwarpMatrix(), (width, height))
    # Combine the result with the original image
    result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
    return result
