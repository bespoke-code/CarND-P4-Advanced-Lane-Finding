import cv2
import numpy as np
import image_manipulation
import calibration_util as calib
import line_util
from moviepy.editor import VideoFileClip
import plots_util

# Global parameters for use
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


def process_image(frame):
    global first_frame
    global width
    global height
    global channels

    global bad_frames_count
    global need_init
    global last_frame_good
    global successive_good_frames
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
    if first_frame:
        width = frame.shape[1]
        height = frame.shape[0]
        channels = frame.shape[2]
        print('Video size information updated.')
        first_frame = False

    # 3. Process the frame using the image processing pipeline.
    #   - undistort image
    #   - filter out the lines from the image
    #   - return a mask to find the lane lines on
    mask = image_manipulation.processFrame(frame)

    # 4. Use the mask to find the lane lines.
    follow_previous_lines = ((bad_frames_count < 3) and (need_init==False))
    #left_line, right_line = line_util.do_line_search(mask, follow_previous_lines)
    left_line, right_line, ploty, left_fitx, right_fitx, left_curverad, dist_center = line_util.do_line_search(mask, True)

    # 5. Sanity check: Are the detected lane lines OK?
    # Performed on the premise that the lines will deviate
    # only a small amount in successive frames.
    # Outliers are neutralized
    #previous_left, previous_right = line_queue.sanitize(left_line, right_line)
    previous_left, previous_right = left_line, right_line

    #if previous_left.detected and previous_right.detected:
    #    successive_good_frames += 1
    #    need_init = False
    #    last_frame_good = True
    #    bad_frames_count = 0
    #else:
    #    last_frame_good = False
    #    successive_good_frames = 0

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
    status_color = (0, 255, 0)
    #overlay = image_manipulation.frameOverlay(frame, previous_left, previous_right, width=frame.shape[1], height=frame.shape[0], color=status_color)
    overlay = image_manipulation.frameOverlay(frame, left_fitx, right_fitx, ploty, left_curverad, dist_center, width=frame.shape[1], height=frame.shape[0], color=status_color)
    return overlay


if __name__ == '__main__':
    out_dir='./data/'
    output = out_dir+'generated_project_video.mp4'

    clip = VideoFileClip("project_video.mp4")
    out_clip = clip.fl_image(process_image)
    # 7. Add frame back to video
    #   - Save Video
    out_clip.write_videofile(output, audio=False)
