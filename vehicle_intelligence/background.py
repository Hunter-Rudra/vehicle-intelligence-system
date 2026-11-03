import numpy as np

class RunningAverageBackground:
    def __init__(self, learning_rate=0.02, threshold=25.0):
        if not 0 < learning_rate <= 1 or threshold < 0:
            raise ValueError("Invalid learning rate or threshold")
        self.learning_rate = float(learning_rate)
        self.threshold = float(threshold)
        self.background = None

    def apply(self, gray):
        frame = np.asarray(gray, dtype=np.float32)
        if frame.ndim != 2:
            raise ValueError("Expected 2D grayscale frame")
        if self.background is None:
            self.background = frame.copy()
            return np.zeros(frame.shape, dtype=bool)
        mask = np.abs(frame - self.background) >= self.threshold
        stable = ~mask
        self.background[stable] = ((1-self.learning_rate)*self.background[stable]
                                    + self.learning_rate*frame[stable])
        return mask
