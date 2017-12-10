from matplotlib import pyplot as plt
import cv2
import numpy as np
import moviepy
import image_manipulation
import calibration_util as calib
import line_util


def side_by_side_plot(image1, image2, title1='Image 1', title2='Image 2'):
    figure, axarr = plt.subplots(1,2)
    axarr[0].imshow(image1)
    axarr[1].imshow(image2)
    plt.show()


def showImage(image, title='Photo'):
    plt.imshow(image)
    plt.title(title)
    plt.show()


def showGrayImage(image, title='Photo'):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


def showChannels(image):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    for c, ax in zip(range(3), axs):
        tmp_img = np.zeros(image.shape, dtype="uint8")
        tmp_img[:, :, c] = image[:, :, c]
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2GRAY)
        ax.imshow(tmp_img, cmap='gray')
        ax.set_axis_off()
    plt.show()


if __name__ == '__main__':
    first_frame = True
    width = 0
    height = 0
    channels = 0

    # Some useful parameters here
    bad_frames_count = 0
    need_init = True
    last_frame_good = False
    successive_good_frames = 0

    # Line queue. Holds line measurements for smoothing out video output
    line_queue = line_util.LineQueue(5)

    # Full processing pipeline
    ###################################################################
    # Get calibration parameters.
    # Full calibration procedure found in calibration_util.py.
    camera_matrix = calib.getCameraCalibration()
    distort_coeffs = calib.getDistortionCoeffs()

    # Enter the video processing loop
    #   - Get video, frame by frame.

    # 1. Grab video frame.
    #   - on the first video frame, grab information about the video size.
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    if first_frame:
        width = frame.shape[1]
        height = frame.shape[0]
        channels = frame.shape[2]
        print('Video size information updated.')

    # 3. Process the frame using the image processing pipeline.
    #   - undistort image
    #   - filter out the lines from the image
    #   - return a mask to find the lane lines on
    mask = image_manipulation.processFrame(frame)

    # 4. Use the mask to find the lane lines.
    follow_previous_lines = ((bad_frames_count < 3) and (need_init==False))
    left_line, right_line = line_util.do_line_search(mask, follow_previous_lines)

    # 5. Sanity check: Are the detected lane lines OK?
    # Performed on the premise that the lines will deviate
    # only a small amount in successive frames.
    # Outliers are neutralized
    previous_left, previous_right = line_queue.sanitize(left_line, right_line)

    if previous_left.detected and previous_right.detected:
        successive_good_frames += 1
        need_init = False
        last_frame_good = True
        bad_frames_count = 0
    else:
        last_frame_good = False
        successive_good_frames = 0

    if follow_previous_lines:
        bad_frames_count = 0
        need_init = False
        last_frame_good = True
        successive_good_frames += 1
    else:
        bad_frames_count += 1
        successive_good_frames = 0
        last_frame_good = False
        if bad_frames_count >= 3:
            need_init = True

    # 6. Draw lines and colour-fill polygon
    #   - Inverse transform to original perspective
    #   - Overlay the polygon of the lane lines
    #   - Measurement status: red (bad, need init), yellow (averaged), green (3 successive frames good)
    if bad_frames_count >= 3 or need_init:
        status_color = (255, 0, 0)
    else:
        if last_frame_good and successive_good_frames >= 3:
            status_color = (0, 255, 0)
        else:
            status_color = (255, 255, 0)
    overlay = image_manipulation.frameOverlay(frame, previous_left, previous_right, width=frame.shape[1], height=frame.shape[0], color=status_color)

    # 7. Add frame back to video
    #   - Save Video
