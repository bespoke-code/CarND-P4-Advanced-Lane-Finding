import src.image_rectification.CameraCalibration as calib
from src.image_rectification import Undistorter
from src.lane_finding.LaneFinder import LaneFinder
from src.lane_finding.LaneSanitizer import LaneSanitizer
from src.lane_finding import ImageProcessor
from moviepy.editor import VideoFileClip
from glob import glob


class VideoProcessorConfig:
    def __init__(self, cameraCalib: calib.Calibration, lane_width_px, lane_length_front_of_car_px):  # TODO: Implement
        assert(cameraCalib is not None)
        self.cameraCalib = cameraCalib
        self.frame_no = 0
        self.first_frame = True
        self.lane_width_px = lane_width_px
        self.lane_length_in_front_of_car_px = lane_length_front_of_car_px

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
    def __init__(self, videoProcConfig: VideoProcessorConfig, undistorter: Undistorter):  # TODO: Implement
        if videoProcConfig is None:
            self.config = VideoProcessorConfig()
        else:
            self.config = videoProcConfig
        self.undistorter = undistorter
        self.frame_no = 0

        self.bad_frames_count = 0
        self.need_init = 0
        self.laneFinder = LaneFinder(lane_width_px=videoProcConfig.lane_width_px,
                                     vertical_search_space_px=videoProcConfig.lane_length_in_front_of_car_px,
                                     num_windows=9
                                     )
        self.laneSanitizer = LaneSanitizer(width=videoProcConfig.width,
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
        mask = image_manipulation.processFrame(frame)  # TODO

        frame = self.undistorter.undistort(frame)
        self.frame_no += 1

        # 3. Use the mask to find the lane lines.
        follow_previous_lines = ((self.bad_frames_count < 4) and (self.need_init is False))



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
