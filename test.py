# A script for various testing purposes.
# Uses code from all scripts
from moviepy.editor import VideoFileClip
from matplotlib import pyplot as plt
import calibration_util as calib
import image_manipulation
import line_util
import cv2
import glob
import numpy as np



# Global parameters for use
first_frame = True
width = 1280
height = 720
channels = 3
i = 0

# Some useful parameters here
bad_frames_count = 0
need_init = True
last_frame_good = False
successive_good_frames = 0

# Line queue. Holds line measurements for smoothing out video output
left_line_queue = line_util.LineSanitizer(720, 1280)
right_line_queue = line_util.LineSanitizer(720, 1280)


# TODO: Incorporate into VideoProcessor?
def saveFrame(frame):
    global i
    print('Saving frame', i, '...')
    plt.imsave(fname='./test_images/frame'+str(i).zfill(5) + '.jpg', arr=frame, format='jpg')
    i += 1
    return frame


def exportFrames(videoFilePath, startSeq, endSeq):
    clip = VideoFileClip(videoFilePath).subclip(startSeq, endSeq)
    out_clip = clip.fl_image(saveFrame)
    output = './data/temp.mp4'
    out_clip.write_videofile(output, audio=False)


def process_frame(frame):
    # pipeline test
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
        print('width:', width)
        print('height:', height)
        print('channels:', channels)

    # 3. Process the frame using the image processing pipeline.
    #   - undistort image
    #   - filter out the lines from the image
    #   - return a mask to find the lane lines on
    mask = image_manipulation.processFrame(frame)

    #plt.imshow(mask, cmap='gray')
    #plt.show()
    #cv2.imshow('Showing frame mask', mask)
    #cv2.waitKey(0)
    # 4. Use the mask to find the lane lines.
    follow_previous_lines = ((bad_frames_count < 2) and (need_init == False))
    #if follow_previous_lines:
    #    print('Following previous lanes.')
    if first_frame:
        #print('Currently at first frame.')
        left_line, right_line, ploty, left_fitx, right_fitx, left_curverad, dist_center = \
            line_util.do_line_search(mask, None, None, True)
        first_frame = False
    else:
        left_line, left_fitx = left_line_queue.get_last(follow_previous_lines)
        right_line, right_fitx = right_line_queue.get_last(follow_previous_lines)

        left_line, right_line, ploty, left_fitx, right_fitx, left_curverad, dist_center = \
            line_util.do_line_search(mask, left_line, right_line, follow_previous_lines) #line_util.do_line_search(mask, left_line, right_line, follow_previous_lines)


    # 5. Sanity check: Are the detected lane lines OK?
    # Performed on the premise that the lines will deviate
    # only a small amount in successive frames.
    # Outliers are neutralized
    left_line_ok = left_line_queue.add(left_line)
    right_line_ok = right_line_queue.add(right_line)

    if left_line_ok and right_line_ok:
        #print('Both lines are OK!')
        successive_good_frames += 1
        need_init = False
        last_frame_good = True
        bad_frames_count = 0
    else:
        #print('One lane is not OK!')
        #print('left OK:', left_line_ok)
        #print('right OK:', right_line_ok)
        last_frame_good = False
        bad_frames_count += 1
        successive_good_frames = 0

    if follow_previous_lines:
        #bad_frames_count = 0
        need_init = False
        last_frame_good = True
        successive_good_frames += 1
    else:
        bad_frames_count += 1
        successive_good_frames = 0
        last_frame_good = False
        if bad_frames_count >= 2:
            need_init = True

    #print('bad frame count:', bad_frames_count)
    if not left_line_ok:
        left_line, left_fitx = left_line_queue.get_last()
    if not right_line_ok:
        right_line, right_fitx = right_line_queue.get_last()

    # 6. Draw lines and colour-fill polygon
    #   - Inverse transform to original perspective
    #   - Overlay the polygon of the lane lines
    #   - TBD: Measurement status: red (bad, need init), yellow (averaged), green (3 successive frames good)
    status_color = (0, 255, 0)
    overlay = image_manipulation.frameOverlay(frame, left_fitx, right_fitx, ploty, left_curverad, dist_center,
                                              width=frame.shape[1], height=frame.shape[0], color=status_color)
    #cv2.imshow('overlay', overlay)
    #cv2.waitKey(0)
    return left_line, right_line

import pandas as pd
if __name__ == '__main__':
    #exportFrames('project_video.mp4', 33, 44)

    left_lines = []
    right_lines = []
    frame_paths = glob.glob('./test_images/frame*')
    frame_paths = sorted(frame_paths)
    print(frame_paths[0], frame_paths[-1])

    X_frames = []

    for frame in frame_paths:
        image = plt.imread(frame)
        X_frames.append(image)

    for ind in np.arange(190, 220):   #len(X_frames)):
        left_line, right_line = process_frame(X_frames[ind])
        #print('Processing frame {num:04d}'.format(num=ind))
        #print('left:', left_line)
        #print('right:', right_line)
        left_lines.append(left_line)
        right_lines.append(right_line)

    for_print = []
    for ind in range(len(left_lines)):
        for_print.append(left_lines[ind][0])

    plt.plot(for_print)
    plt.show()


