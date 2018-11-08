import argparse

class CLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("calib", help="Calibration file for your camera")
        self.parser.add_argument("--no-undistort", help="Do not undistort the image")
