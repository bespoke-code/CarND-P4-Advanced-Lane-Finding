from src.image_rectification.CameraCalibration import Calibration, CalibrationFactory
from src.image_rectification.Undistorter import Undistorter
from src.video_processing.VideoProcessor import VideoProcessor, VideoProcessorConfig

if __name__ == '__main__':
    camera_calibration = CalibrationFactory.SamsungGalaxyS7Calibration1440p()
    undistorter = Undistorter(camera_calibration)
    # TODO
    videoConfig = VideoProcessorConfig(camera_calibration)
    videoProc = VideoProcessor(videoProcConfig=videoConfig)

    filename = "VideoFileName"
    outputFolder = "OutputFolder"
    videoProc.processVideo(filename, outputFolder)
