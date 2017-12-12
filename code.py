import cv2
import numpy as np
import image_manipulation
import calibration_util as calib
import line_util
from moviepy.editor import VideoFileClip
import plots_util

# Global parameters for use
first_frame = True
width = 1280
height = 720
channels = 3

# Some useful parameters here
bad_frames_count = 0
need_init = True
last_frame_good = False
successive_good_frames = 0

# Line queue. Holds line measurements for smoothing out video output
left_line_queue = line_util.LineSanitizer(720, 1280)
right_line_queue = line_util.LineSanitizer(720, 1280)


def process_image(frame):
    global first_frame
    global width
    global height
    global channels

    global bad_frames_count
    global need_init
    global last_frame_good
    global successive_good_frames
    global left_line_queue
    global right_line_queue
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

    # 3. Process the frame using the image processing pipeline.
    #   - undistort image
    #   - filter out the lines from the image
    #   - return a mask to find the lane lines on
    mask = image_manipulation.processFrame(frame)

    # 4. Use the mask to find the lane lines.
    follow_previous_lines = ((bad_frames_count < 2) and (need_init==False))

    if first_frame:
        left_line, right_line, ploty, left_fitx, right_fitx, left_curverad, dist_center = \
            line_util.do_line_search(mask, None, None, True)
        first_frame = False
    else:
        left_line, left_fitx = left_line_queue.get_last(follow_previous_lines)
        right_line, right_fitx = right_line_queue.get_last(follow_previous_lines)

        left_line, right_line, ploty, left_fitx, right_fitx, left_curverad, dist_center = \
            line_util.do_line_search( mask, left_line, right_line, follow_previous_lines)

    # 5. Sanity check: Are the detected lane lines OK?
    # Performed on the premise that the lines will deviate
    # only a small amount in successive frames.
    # Outliers are neutralized
    left_line_ok = left_line_queue.add(left_line)
    right_line_ok = right_line_queue.add(right_line)

    if left_line_ok and right_line_ok:
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
        if bad_frames_count >= 2:
            need_init = True

    if not left_line_ok:
        left_line, left_fitx = left_line_queue.get_last()
    if not right_line_ok:
        right_line, right_fitx = right_line_queue.get_last()

    # 6. Draw lines and colour-fill polygon
    #   - Inverse transform to original perspective
    #   - Overlay the polygon of the lane lines
    #   - TBD: Measurement status: red (bad, need init), yellow (averaged), green (3 successive frames good)
    status_color = (0, 255, 0)
    overlay = image_manipulation.frameOverlay(frame, left_fitx, right_fitx, ploty, left_curverad, dist_center, width=frame.shape[1], height=frame.shape[0], color=status_color)
    return overlay


if __name__ == '__main__':
    out_dir='./data/'
    output = out_dir+'generated_project_video.mp4'

    clip = VideoFileClip("project_video.mp4").subclip(33,44)
    out_clip = clip.fl_image(process_image)
    # 7. Add frame back to video
    #   - Save Video
    out_clip.write_videofile(output, audio=False)
