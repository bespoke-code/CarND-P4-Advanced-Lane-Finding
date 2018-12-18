import cv2
import numpy as np


class ImageWarper:
    def __init__(self, org_tl, org_tr, org_bl, org_br, dst_tl, dst_tr, dst_bl, dst_br, width, height):
        src_points = np.float32((org_tl, org_tr, org_bl, org_br))
        dst_points = np.float32((
            dst_tl,  # top left
            dst_tr,  # top right
            dst_bl,  # bottom left
            dst_br   # bottom right
        ))

        self.width = width
        self.height = height

        self.lane_len_in_front_px = np.ceil(np.abs(dst_tl - dst_bl))
        self.warped_bottom_left = dst_bl
        self.warped_bottom_right = dst_br

        self.warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.unwarp_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

    def warp(self, image: np.array) -> np.array:
        assert (self.width == image.shape[1])
        assert (self.height == image.shape[0])
        return cv2.warpPerspective(image, self.warp_matrix, (self.width, self.height))

    def unwarp(self, image: np.array) -> np.array:
        assert (self.width == image.shape[1])
        assert (self.height == image.shape[0])
        return cv2.warpPerspective(image, self.unwarp_matrix, (self.width, self.height))

    @property
    def lane_width_px(self):
        return np.ceil(np.abs(self.warped_bottom_left[0]-self.warped_bottom_right[1]))

    @classmethod
    def BirdsEyeWarp(cls, org_tl, org_tr, org_bl, org_br, width, height):
        top_padding_coef = 2 / 9
        left_padding_coef = 6.5 / 16
        right_padding_coef = 6.5 / 16
        bottom_padding_coef = 0 / 9

        warped_top_left_pt = (width * left_padding_coef, height * top_padding_coef)
        warped_top_right_pt = (width * (1. - right_padding_coef), height * top_padding_coef)
        warped_bottom_left_pt = (width * left_padding_coef, height * (1. - bottom_padding_coef))
        warped_bottom_right_pt = (width * (1. - right_padding_coef), height * (1. - bottom_padding_coef))

        return ImageWarper(org_tl=org_tl,
                           org_tr=org_tr,
                           org_bl=org_bl,
                           org_br=org_br,
                           dst_tl=warped_top_left_pt,
                           dst_tr=warped_top_right_pt,
                           dst_bl=warped_bottom_left_pt,
                           dst_br=warped_bottom_right_pt,
                           width=width,
                           height=height
                           )


class ImageProcessor:
    def sobel(self, image: np.array, direction):
        if direction == 'x':  # Sobel X
            return cv2.Sobel(image, cv2.CV_64F, 1, 0)
        else:
            return cv2.Sobel(image, cv2.CV_64F, 0, 1)

    # From the Udacity Advanced Lane Finding lectures
    def abs_sobel_thresh(self, img: np.array, orient='x', sobel_kernel=3, thresh=(0, 255)) -> np.array:
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

    def mag_threshold(self, img: np.array, sobel_kernel=3, mag_thresh=(0, 255)) -> np.array:
        """
        Applies Sobel x and y, then computes the magnitude of
        the gradient and applies a threshold.
        :param img: The image to apply the filter to.
        :param sobel_kernel: Kernel size
        :param mag_thresh: Gradient magnitude threshold
        :return: The image after the filter is applied.
        """
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

    def dir_threshold(self, img: np.array, sobel_kernel=3, thresh=(0, np.pi / 2)) -> np.array:
        """
        Applies Sobel x and y, then computes the direction of
        the gradient and applies a threshold.
        :param img: The image to apply the filter to.
        :param sobel_kernel: Kernel size
        :param thresh: Directional gradient threshold
        :return: The image after the filter is applied.
        """
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

    def sobel_select(self, img: np.array, thresh=(20, 100)) -> np.array:
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

    # Color/gradient thresholding
    #######################################################################
    def lines_CMYK(self, image):
        if np.max(image[:, :, 2]) > 1:
            image = image / 255.

        # As seen at https://www.rapidtables.com/convert/color/rgb-to-cmyk.html
        b = image[:, :, 2]
        k = 1 - image.max(axis=2)
        y = (1.000001 - b - k) / (1.000001 - k)

        y_select = (y * 255.) > 90.
        y_select_false_positive = (y * 255.) >= 250.
        k_select = (k * 255.) < 45.

        yk_mask = np.zeros_like(k, dtype=np.uint8)
        yk_mask[y_select] = 255
        yk_mask[y_select_false_positive] = 0
        yk_mask[k_select] = 255

        return yk_mask

    # Yellow lane lines extraction
    def yellow_lines_RGB(self, image):
        color_select_img = np.copy(image)
        red_threshold = 220
        green_threshold = 180
        blue_threshold = 40

        # Identify pixels below the threshold. Black 'em out
        colour_thresholds = (image[:, :, 0] < red_threshold) | \
                            (image[:, :, 1] < green_threshold) | \
                            (image[:, :, 2] < blue_threshold)
        color_select_img[colour_thresholds] = [0, 0, 0]
        color_select_img = cv2.cvtColor(color_select_img, cv2.COLOR_RGB2GRAY)
        slice = color_select_img[:, :] > 0
        color_select_img[slice] = 255
        return color_select_img

    # White lane lines extraction
    def white_lines_RGB(self, image):
        color_select_img = np.copy(image)
        red_threshold = 200
        green_threshold = 185
        blue_threshold = 199

        # Identify pixels below the threshold. Black 'em out
        colour_thresholds = (image[:, :, 0] < red_threshold) | \
                            (image[:, :, 1] < green_threshold) | \
                            (image[:, :, 2] < blue_threshold)
        color_select_img[colour_thresholds] = [0, 0, 0]
        color_select_img = cv2.cvtColor(color_select_img, cv2.COLOR_RGB2GRAY)
        slice = color_select_img[:, :] > 0
        color_select_img[slice] = 255
        return color_select_img

    def combined_sobel(self, img, ksize=3, thresh=(0, 255), mag_thresh=(0, 255), dir_thresh=(0, np.pi / 2)):
        # Choose a larger k-size (odd number) to get smoother gradient measurements
        # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=thresh)
        grady = self.abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=thresh)
        mag_binary = self.mag_threshold(img, sobel_kernel=ksize, mag_thresh=mag_thresh)
        dir_binary = self.dir_threshold(img, sobel_kernel=ksize, thresh=dir_thresh)

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined

    # Create mask
    def mask(self, image):
        """
        Image processing pipeline for a video frame / standalone road image.
        :param image: A video frame or an RGB camera image.
        :return: A mask showing the lane lines on the road image.
        """
        kernel = np.ones((3, 3), np.uint8)

        color_mask = cv2.add(self.yellow_lines_RGB(image), self.white_lines_RGB(image))
        color_mask[color_mask > 0] = 255
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        sobel_mask = cv2.add(
            self.combined_sobel(
                cv2.blur(image, (9, 9)), ksize=3,
                thresh=(15, 90),
                mag_thresh=(10, 90),
                dir_thresh=(0.8, 1.)
            ),
            self.combined_sobel(
                cv2.blur(image, (9, 9)), ksize=3,
                thresh=(15, 90),
                mag_thresh=(10, 90),
                dir_thresh=(0.01, 0.2)
            )
        )
        sobel_mask[sobel_mask > 0] = 255

        sobel_mask = cv2.erode(sobel_mask, kernel, iterations=1)
        sobel_mask = cv2.morphologyEx(sobel_mask, cv2.MORPH_OPEN, kernel)
        cmyk_mask = self.lines_CMYK(image)

        result_mask = np.zeros_like(color_mask)
        result_mask[sobel_mask > 0] = 255
        result_mask[color_mask > 0] = 255
        result_mask[cmyk_mask > 0] = 255
        return result_mask

    def overlayPolygon_warped(self, frame, left_fitx_p, right_fitx_p, ploty, left_curvature, distance, width, height, color=(0, 255, 0)):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(frame[:, :, 1]).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx_p, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx_p, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), color)

        return color_warp

    def combineOverlay(self, image, overlay):
        # Combine the result with the original image

        result = cv2.addWeighted(image, 1, overlay, 0.5, 0)
        #font = cv2.FONT_HERSHEY_PLAIN
        #cv2.putText(result, 'Left lane radius: ' + str(left_curvature) + 'm', (30, 50), font, 1.7, (255, 255, 255), 2,
        #            cv2.LINE_AA)
        #cv2.putText(result, 'Distance to lane center: ' + str(distance * 100) + 'cm', (30, 80), font, 1.7,
        #            (255, 255, 255),
        #            2, cv2.LINE_AA)
        return result