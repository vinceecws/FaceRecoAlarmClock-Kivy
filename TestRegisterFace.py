from multiprocessing import Process, Queue
from FaceRecognitionAPI import FaceRecognitionAPI
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as f
import time
import cv2
import os
import numpy as np

def update(self, dt):
    _, frame = self.capture.read()
    frame = self.detectFace(frame)
    buf1 = cv2.flip(frame, 0)
    buf = buf1.tostring()
    texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    self.img1.texture = texture1

def initializeFaceDetection(self):
    self.facerecognition = FaceRecognitionAPI(self.face_dir, self.weight_dir, self.haar_dir)
    self.capture = cv2.VideoCapture(0)
    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320);
    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240);
    self.model = Siamese_MobileNetV2()
    self.model.eval()

def detectFace(self, frame):
    detected, processed_frame, image = self.facerecognition.preprocessFrame(frame)
    if detected:
        if self.p:
            if self.p.is_alive():
                pass
            else:
                retval = self.queue.get()
                print(retval)
                self.p = None
        else:
            self.p = Process(target=self.identifyFace, args=(self.queue, image, self.face_vector,))
            self.p.start()

    return processed_frame 

def main(args):

if __name__ == '__main__':
	args = None
	main(args)