from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from multiprocessing import Process, Queue
from Siamese_MobileNetV2.Siamese_MobileNetV2 import Siamese_MobileNetV2
from FaceRecognitionAPI import FaceRecognitionAPI
from torchvision import transforms
from PIL import Image as PImage
import torch
import torch.nn.functional as f
import time
import cv2
import os
import numpy as np

class CamApp(App):

    def build(self):
        #Directories
        self.face_dir = './faces'
        self.weight_dir = './Siamese_MobileNetV2/weights/siamese_mobilenet_v2_pretrained_500.pkl'
        self.haar_dir = './Siamese_MobileNetV2/haarcascade_frontalface_default.xml'

        self.p = None
        self.img1 = Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        self.queue = Queue()
        self.initializeFaceDetection()

        self.face_vector = self.facerecognition.loadFace('')

        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def on_request_close(self, *args):
        self.capture.release()
        return True

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

    def identifyFace(self, queue, image, face_id):
        '''
        output = self.model(image).detach().numpy()
        queue.put(output)
        res = self.isSame(queue, label)
        queue.put(res)
        '''
        res = self.facerecognition.runRecognition(image, face_id)
        queue.put(res)


if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()