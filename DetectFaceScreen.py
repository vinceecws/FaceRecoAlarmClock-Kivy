from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from multiprocessing import Process, Queue
from FaceRecognitionAPI import FaceRecognitionAPI
from torchvision import transforms
from PIL import Image as PImage
import torch
import time
import cv2
import os
import numpy as np

class CamApp(App):

    def build(self):
        #Directories
        self.face_dir = './faces'
        self.weight_dir = './MobileFaceNet_Pytorch/model/best/068.ckpt'
        self.haar_dir = './Siamese_MobileNetV2/src/utils/haarcascade_frontalface_default.xml'

        self.p = None
        self.img1 = Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        self.queue = Queue()
        self.initializeFaceDetection()

        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def on_request_close(self, *args):
        self.capture.release()
        return True

    def update(self, dt):
        _, frame = self.capture.read()
        frame = self.mirrorizeFrame(frame)
        frame = self.detectFace(frame)
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = texture1

    def initializeFaceDetection(self):
        self.facerecognition = FaceRecognitionAPI(self.face_dir, self.weight_dir, self.haar_dir)
        self.capture = cv2.VideoCapture(1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 350);
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 350);

    def detectFace(self, frame):
        detected, processed_frame, image, original = self.facerecognition.cropAndPreprocessFrame(frame)
        if detected:
            if self.p:
                if self.p.is_alive():
                    pass
                else:
                    retval = self.queue.get()
                    print(retval)
                    self.p = None
            else:
                self.p = Process(target=self.identifyFace, args=(self.queue, image,))
                self.p.start()

        return processed_frame

    def identifyFace(self, queue, image):
        res = self.facerecognition.runRecognition(image)
        queue.put(res)

    def mirrorizeFrame(self, frame):
        return cv2.flip(frame, 1)

if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()