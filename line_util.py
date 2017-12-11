import numpy as np
from matplotlib import pyplot as plt
import cv2
import matplotlib.image as mpimg
from queue import Queue

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # TODO: Define these. See image_manip polygon fill for more
        self.ploty = []
        self.fitx = []


class LineQueue():
    def __init__(self, capacity=5):
        self.left_line_queue = Queue(capacity)
        self.right_line_queue = Queue(capacity)
        self.capacity = capacity

        self.last_right_OK = False
        self.last_left_OK = False

    def last_good_left_line(self):
        # TODO: calculate last good left measurement.
        pass

    def last_good_right_line(self):
        # TODO: calculate last good left measurement.
        pass

    def add_line(self, newLine, isLeft=True):
        if isLeft:
            self.left_line_queue.put(newLine)
        else:
            self.right_line_queue.put(newLine)

    def sanitize(self, left_line, right_line):
        pass
        # TODO: finish code here. check if deviation is too big (use arctg and curve radius)
        # Left line
        criteria_pass_left = True
        # Right line
        criteria_pass_right = True

        if criteria_pass_left:
            # Line is OK!
            self.add_line(left_line, True)

        if criteria_pass_right:
            self.add_line(right_line, False)

        return self.last_good_left_line(), self.last_good_right_line()


def do_line_search(image_mask, follow_previous_lines):
    # Udacity code here, adapted to fit the project's needs

    if not follow_previous_lines:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(image_mask[np.int(image_mask.shape[0] / 2):, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((image_mask, image_mask, image_mask)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(image_mask.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 100
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image_mask.shape[0] - (window + 1) * window_height
            win_y_high = image_mask.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
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
    else:
        nonzero = image_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                          (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                           (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    #print('left line:', left_fit)
    #print('right line:', right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, image_mask.shape[0] - 1, image_mask.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30. / 720.  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 640.  # meters per pixel in x dimension

    # See if the curvatures align now!
    # Define conversions in x and y from pixels space to meters
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / \
                    np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / \
                     np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters.

    left_line_bottom = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_line_bottom = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

    image_center = 1280 / 2
    lane_center = int((right_line_bottom - left_line_bottom) / 2. + left_line_bottom)
    dist_center = lane_center - image_center
    dist_center *= xm_per_pix  # in meters. Negative numbers represent our vehicle being closer to the right lane line,
                               # while positive numbers represent the vehicle deviating to the left of the lane.

    # TODO: Use line class to return left and right lines
    return left_fit, right_fit, ploty, left_fitx, right_fitx, left_curverad, dist_center
    #return Line(left_fit, left_fitx, ), Line()
