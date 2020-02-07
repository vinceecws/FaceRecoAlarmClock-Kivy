import numpy as np
import cv2

class OpticalFlowController():

    def __init__(self, frame_width=320, frame_height=240):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width);
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height);

    def horizontalSpeedLimit(self, flow, limit=6.0):
        return np.max(flow[:,:,0]) > limit

    def verticalSpeedLimit(self, flow, limit=6.0):
        return np.max(flow[:,:,1]) > limit

    def step(self, prev=None):
        _, frame = self.cap.read()
        frame = self.flipFrame(frame)

        if prev is None:
            prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame = self.cap.read()
            frame = self.flipFrame(frame)

        nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        if self.horizontalSpeedLimit(flow) or self.verticalSpeedLimit(flow):
            return frame, nxt, True

        return frame, nxt, False

    def release(self):
        self.cap.release()

    def flipFrame(self, frame):
        return cv2.flip(frame, 1)