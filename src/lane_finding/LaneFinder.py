import numpy as np


class LaneFinder():
    def __init__(self, lane_width_px, vertical_search_space_px, num_windows):
        self.minimum_pixels_found = 100  # px
        self.window_margin = 100  # px
        self.lane_expected_width_px = lane_width_px
        self.region_len_in_front_px = vertical_search_space_px
        self.num_windows = num_windows

    def do_line_search(self, image_mask, known_left_fit=None, known_right_fit=None, follow_previous_lines=False):
        # Udacity code here, adapted to fit the project's needs

        image_width = image_mask.shape[1]
        image_height = image_mask.shape[0]

        # Check to avoid stupidity
        if known_left_fit is None or known_right_fit is None and follow_previous_lines:
            follow_previous_lines = False

        if not follow_previous_lines:
            # Take a histogram of the bottom half of the image
            histogram = np.sum(image_mask[np.int(image_height / 2):, :], axis=0)
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((image_mask, image_mask, image_mask)) * 255
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = self.num_windows
            # Set height of windows
            window_height = np.int(image_height / nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = image_mask.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Set the width of the windows +/- margin
            margin = self.window_margin
            # Set minimum number of pixels found to recenter window
            minpix = self.minimum_pixels_found
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = image_height - (window + 1) * window_height
                win_y_high = image_height - window * window_height
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
            left_lane_inds = ((nonzerox > (
            known_left_fit[0] * (nonzeroy ** 2) + known_left_fit[1] * nonzeroy + known_left_fit[2] - margin)) &
                              (nonzerox < (
                              known_left_fit[0] * (nonzeroy ** 2) + known_left_fit[1] * nonzeroy + known_left_fit[
                                  2] + margin)))

            right_lane_inds = ((nonzerox > (
            known_right_fit[0] * (nonzeroy ** 2) + known_right_fit[1] * nonzeroy + known_right_fit[2] - margin)) &
                               (nonzerox < (
                               known_right_fit[0] * (nonzeroy ** 2) + known_right_fit[1] * nonzeroy + known_right_fit[
                                   2] + margin)))

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # print('left line:', left_fit)
        # print('right line:', right_fit)

        # Generate x and y values for plotting
        ploty = np.linspace(0, image_height - 1, image_height)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30. / self.region_len_in_front_px  # meters per pixel in y dimension
        xm_per_pix = 3.7 / self.lane_expected_width_px  # meters per pixel in x dimension

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

        image_center = image_width / 2
        lane_center = int((right_line_bottom - left_line_bottom) / 2. + left_line_bottom)
        dist_center = lane_center - image_center
        dist_center *= xm_per_pix  # in meters. Negative numbers represent our vehicle being closer to the right lane line,
        # while positive numbers represent the vehicle deviating to the left of the lane.

        return left_fit, right_fit, ploty, left_fitx, right_fitx, np.round(left_curverad, 3), np.round(dist_center, 4)
