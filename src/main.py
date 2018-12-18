from src.image_rectification.CameraCalibration import CalibrationFactory
from src.image_rectification.Undistorter import Undistorter
from src.video_processing.VideoProcessor import VideoProcessor, VideoProcessorConfig
from src.lane_finding.ImageProcessor import ImageWarper, ImageProcessor


if __name__ == '__main__':
    # trapeze
    top_left_pt = (561,469)
    top_right_pt = (680,469)
    bottom_left_pt = (236,711)
    bottom_right_pt = (1156,711)

    camera_calibration = CalibrationFactory.SamsungGalaxyS7Calibration1440p()
    undistorter = Undistorter(camera_calibration)
    warper = ImageWarper.BirdsEyeWarp(org_tl=top_left_pt,
                                      org_tr=top_right_pt,
                                      org_bl=bottom_left_pt,
                                      org_br=bottom_right_pt,
                                      width=camera_calibration.width,
                                      height=camera_calibration.height
                                      )
    processor = ImageProcessor()

    videoConfig = VideoProcessorConfig(cameraCalib=camera_calibration,
                                       warper=warper
                                       )
    videoProc = VideoProcessor(videoProcConfig=videoConfig,
                               undistorter=undistorter,
                               imgProc=processor
                               )

    filename = "VideoFileName"
    outputFolder = "OutputFolder"
    videoProc.processVideo(filename, outputFolder)
