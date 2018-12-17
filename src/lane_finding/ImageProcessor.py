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
    def __init__(self):
        pass

    @staticmethod
    def _sobel(image: np.array, direction):
        if direction == 'x':  # Sobel X
            return cv2.Sobel(image, cv2.CV_64F, 1, 0)
        else:
            return cv2.Sobel(image, cv2.CV_64F, 0, 1)

    # From the Udacity Advanced Lane Finding lectures
    @staticmethod
    def _abs_sobel_thresh(img: np.array, orient='x', sobel_kernel=3, thresh=(0, 255)) -> np.array:
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

    @staticmethod
    def _mag_threshold(img: np.array, sobel_kernel=3, mag_thresh=(0, 255)) -> np.array:
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

    @staticmethod
    def _dir_threshold(img: np.array, sobel_kernel=3, thresh=(0, np.pi / 2)) -> np.array:
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

    @staticmethod
    def _sobel_select(img: np.array, thresh=(20, 100)) -> np.array:
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
