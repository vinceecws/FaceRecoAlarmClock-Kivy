from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from FaceRecognitionAPI import FaceRecognitionAPI
from OpticalFlowController import OpticalFlowController
import time
import cv2
import os
import numpy as np

class CamApp(App):

    def build(self):
        Window.bind(on_request_close=self.on_request_close)
        #Directories
        self.face_dir = './faces'
        self.weight_dir = './Siamese_MobileNetV2/weights/siamese_mobilenet_v2_pretrained_500.pkl'
        self.haar_dir = './Siamese_MobileNetV2/haarcascade_frontalface_default.xml'

        self.prev = None
        self.screencaps = []
        self.stream = Image()
        layout = BoxLayout()
        layout.add_widget(self.stream)
        self.initializeOpticalFlow()

        Clock.schedule_interval(self.update, 1.0/33.0)
        Clock.schedule_interval(self.capture, 1.0/2.0) #2FPS
        return layout

    def on_request_close(self, *args, **kwargs):
        self.opticalflowcontroller.release()
        face_id = self.facerecognition.processAndRegister(self.screencaps)
        return True

    def update(self, dt): #dt is the time since the last interval?
        self.frame, self.prev, overLimit = self.opticalflowcontroller.step(self.prev)
        if overLimit:
            print('Too fast')

        buf = cv2.flip(self.frame, 0)
        buf = buf.tostring()
        streamTexture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        streamTexture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.stream.texture = streamTexture

    def capture(self, dt): #capture every 0.5 s
        print('Capture')
        self.screencaps.append(self.frame)

    def initializeOpticalFlow(self):
        self.facerecognition = FaceRecognitionAPI(self.face_dir, self.weight_dir, self.haar_dir)
        self.opticalflowcontroller = OpticalFlowController()

if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()