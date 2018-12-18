import src.image_rectification.CameraCalibration as calib
from src.image_rectification import Undistorter
from src.lane_finding.LaneFinder import LaneFinder
from src.lane_finding.LaneSanitizer import LaneSanitizer
from src.lane_finding.ImageProcessor import ImageProcessor, ImageWarper
from moviepy.editor import VideoFileClip
from glob import glob


class VideoProcessorConfig:
    def __init__(self, cameraCalib: calib.Calibration, warper: ImageWarper):  # TODO: Implement
        assert(cameraCalib is not None)
        self.cameraCalib = cameraCalib
        self.imgWarper = warper
        self.frame_no = 0
        self.first_frame = True

    @property
    def lane_width_px(self):
        return self.imgWarper.lane_width_px

    @property
    def lane_length_in_front_of_car_px(self):
        return self.imgWarper.lane_len_in_front_px

    @property
    def width(self):
        return self.cameraCalib.width

    @property
    def height(self):
        return self.cameraCalib.height

    @property
    def is_firstFrame(self):
        return 0 == self.frame_no


class VideoProcessor:
    def __init__(self, videoProcConfig: VideoProcessorConfig, undistorter: Undistorter, imgProc: ImageProcessor):  # TODO: Implement
        assert(videoProcConfig is not None)

        self.config = videoProcConfig
        self.undistorter = undistorter
        self.imgProc = imgProc
        self.frame_no = 0

        self.bad_frames_count = 0
        self.need_init = 0
        self.laneFinder = LaneFinder(lane_width_px=videoProcConfig.lane_width_px,
                                     vertical_search_space_px=videoProcConfig.lane_length_in_front_of_car_px,
                                     num_windows=9
                                     )
        self.laneSanitizer_left = LaneSanitizer(width=videoProcConfig.width,
                                                height=videoProcConfig.height
                                               )
        self.laneSanitizer_right = LaneSanitizer(width=videoProcConfig.width,
                                                 height=videoProcConfig.height
                                                )

    @property
    def first_frame(self):
        return self.frame_no == 0

    def process_frame(self, frame):
        # 1. Grab video frame.
        #   - on the first video frame, grab information about the video size.
        if self.first_frame:
            assert(self.config.width == frame.shape[1])
            assert(self.config.height == frame.shape[0])
            print('Video size information confirmed.')

        # 2. Process the frame using the image processing pipeline.
        #   - undistort image
        #   - filter out the lines from the image
        #   - return a mask to find the lane lines on
        frame = self.undistorter.undistort(frame)
        self.frame_no += 1
        warped_frame = self.config.imgWarper.warp(frame)

        mask = self.imgProc.mask(warped_frame)
        # 3. Use the mask to find the lane lines.
        follow_previous_lines = ((self.bad_frames_count < 4) and (self.need_init is False))

        if self.first_frame:
            left_line, right_line, ploty, left_fitx, right_fitx, left_curverad, dist_center = \
                self.laneFinder.do_line_search(mask, None, None, False)
        else:
            left_line, left_fitx = self.laneSanitizer_left.get_last(follow_previous_lines)
            right_line, right_fitx = self.laneSanitizer_right.get_last(follow_previous_lines)

            left_line, right_line, ploty, left_fitx, right_fitx, left_curverad, dist_center = \
                self.laneFinder.do_line_search(mask, left_line, right_line, follow_previous_lines)

        # 5. Sanity check: Are the detected lane lines OK?
        # Performed on the premise that the lines will deviate
        # only a small amount in successive frames.
        # Outliers are neutralized
        left_line_ok = self.laneSanitizer_left.add(left_line)
        # left_line_ok = True
        right_line_ok = self.laneSanitizer_right.add(right_line)
        # right_line_ok = True

        if left_line_ok and right_line_ok:
            self.successive_good_frames += 1
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
            if bad_frames_count >= 5:
                need_init = True

        if not left_line_ok:
            left_line, left_fitx = self.laneSanitizer_left.get_last()
        if not right_line_ok:
            right_line, right_fitx = self.laneSanitizer_right.get_last()

        # 6. Draw lines and colour-fill polygon
        #   - Inverse transform to original perspective
        #   - Overlay the polygon of the lane lines
        #   - TBD: Measurement status: red (bad, need init), yellow (averaged), green (3 successive frames good)
        status_color = (0, 255, 0)
        overlay_warped = self.imgProc.overlayPolygon_warped(frame,
                                                            left_fitx,
                                                            right_fitx,
                                                            ploty,
                                                            left_curverad,
                                                            dist_center,
                                                            width=self.config.width,
                                                            height=self.config.height,
                                                            color=status_color)


        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        overlay_unwarped = self.config.imgWarper.unwarp(overlay_warped)
        return self.imgProc.combineOverlay(frame, overlay_unwarped)


    def processVideo(self, filename: str, outputFolder: str, outputFilename: str):
        clip = VideoFileClip(filename) #.subclip(35,43)
        out_clip = clip.fl_image(self.process_frame)
        # 7. Add frame back to video
        #   - Save Video
        if outputFolder[-1] != '/':
            output = outputFolder + '/' + outputFilename
        else:
            output = outputFolder + outputFilename
        print('Processing video {name}...'.format(name=filename))
        out_clip.write_videofile(output, audio=False)

    def processFolder(self, inputFolder: str, outputFolder: str):
        if inputFolder[-1] != '/':
            inputFolder = inputFolder + '/'

        folder = glob(inputFolder + '*.mp4')
        for video in folder:
            filetype = video.split('/')[-1].split('.')[-1]
            outputFilename = video.split('/')[-1].split('.')[0] + '_processed.' + filetype
            self.processVideo(video, outputFolder, outputFilename)
