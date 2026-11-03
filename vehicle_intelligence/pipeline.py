from .preprocessing import to_grayscale, gaussian_blur
from .background import RunningAverageBackground
from .components import connected_components
from .tracking import CentroidTracker
from .counting import LineCounter

class VehiclePipeline:
    def __init__(self, min_area=30, motion_threshold=22.0, line_y=100.0):
        self.background = RunningAverageBackground(threshold=motion_threshold)
        self.tracker = CentroidTracker()
        self.counter = LineCounter(line_y)
        self.min_area = int(min_area)

    def process(self, frame):
        gray = to_grayscale(frame)
        smooth = gaussian_blur(gray, size=3, sigma=0.8)
        mask = self.background.apply(smooth)
        detections = connected_components(mask, self.min_area)
        tracks = self.tracker.update(detections)
        return {"motion_mask":mask, "detections":detections,
                "tracks":tracks, "crossings":self.counter.update(tracks)}
