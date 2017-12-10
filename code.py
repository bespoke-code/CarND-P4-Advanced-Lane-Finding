from matplotlib import pyplot as plt
import cv2
import numpy as np
import moviepy
import image_manipulation
import calibration_util as calib
import glob


def side_by_side_plot(image1, image2, title1='Image 1', title2='Image 2'):
    figure, axarr = plt.subplots(1,2)
    axarr[0].imshow(image1)
    axarr[1].imshow(image2)
    plt.show()


def find_laneLines_sobel(img, ksize=3, thresh=(0,255), mag_thresh=(0,255), dir_thresh=(0, np.pi/2)):
    # Choose a larger k-size (odd number) to get smoother gradient measurements

    # Apply each of the thresholding functions
    gradx = image_manipulation.abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=thresh)
    grady = image_manipulation.abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=thresh)
    mag_binary = image_manipulation.mag_thresh(img, sobel_kernel=ksize, mag_thresh=mag_thresh)
    dir_binary = image_manipulation.dir_threshold(img, sobel_kernel=ksize, thresh=dir_thresh)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined

# Transform to a birds-eye perspective
def perspective_transform_matrix(input_points, output_points):
    return cv2.getPerspectiveTransform(input_points, output_points)


def perspective_transform(image, perspective_matrix):
    return cv2.warpPerspective(image, perspective_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)


def find_lane_lines(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

if __name__ == '__main__':
    # Full processing pipeline here

    # Preparations
    calib_folder = './camera_cal/'
    calib_files = glob.glob(calib_folder + 'calibration*.jpg')

    # Calibrate camera
    camera_matrix, distort_coeffs = calib.getCalibParams(calib_files)
    # Undistort frame
    random_ind = np.random.randint(0, len(calib_files))

    #original_image = plt.imread(calib_files[random_ind])
    original_image = plt.imread('./test_images/straight_lines1.jpg')
    width = original_image.shape[1]
    height = original_image.shape[0]
    print(height, width)
    undistorted_image = cv2.undistort(original_image, camera_matrix, distort_coeffs, None, newCameraMatrix=camera_matrix)

    # Show original and undistorted image
    side_by_side_plot(original_image, undistorted_image)

    # Implement color & gradient threshold
    # White lane detect
    # Yellow lanes detect
    # Color/gradient threshold
    #color_grad_thresh = image_manipulation.color_and_gradient_threshold(undistorted_image)

    # Warp image (perspective transform)
    #top_left = [520, 500] # [7/16*width, 6/10*height]
    top_left = [594, 450] # [7/16*width, 6/10*height]
    #top_right = [768, 500] # [9/16*width, 6/10*height]
    top_right = [687, 450] # [9/16*width, 6/10*height]
    bottom_left = [262, 670] # [1/16*width, 9/10*height]
    bottom_right = [1044, 670] #[15/16*width, 9/10*height]
    src_points = np.float32([top_left, top_right, bottom_left, bottom_right])
    dst_points = np.float32([
        [3/16*width, 3/10*height], # top left
        [13/16*width, 3/10*height], # top right
        [3/16*width, 10/10*height], # bottom left
        [13/16*width, 10/10*height]  # bottom right
    ])
    print(src_points)
    print(dst_points)

    warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    unwarp_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

    warped_image = cv2.warpPerspective(undistorted_image, warp_matrix, (width, height))

    side_by_side_plot(undistorted_image, warped_image)
    # Find lane curvature
    # draw lines and colour-fill polygon
    # Inverse transform to original perspective
    # Add frame to video
    # Save Video
