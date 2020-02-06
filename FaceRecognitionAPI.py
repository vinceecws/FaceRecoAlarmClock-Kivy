import os
import torch
import cv2
import uuid
from PIL import Image
from torchvision import transforms
from Siamese_MobileNetV2.Siamese_MobileNetV2 import Siamese_MobileNetV2

class FaceRecognitionAPI():
    def __init__(self, face_dir, weight_dir, haar_dir):
        assert os.path.isdir(face_dir), "{} is not a folder.".format(face_dir)
        self.faces = os.listdir(face_dir)

        #Initialize Haar
        self.face_cascade = cv2.CascadeClassifier(haar_dir)

        #Initialize model
        self.input_width = 224
        self.input_height = 224
        self.model = Siamese_MobileNetV2()
        self.model.eval()
        self.loadModelWeights(weight_dir)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def registerFace(self, current_id=None):
        face_vector = self.runRecognition()
        if not current_id:
            current_id = str(uuid.uuid4()) #Generate new id

        self.writeFace(face_vector, current_id) #Add new entry

    def loadFace(self, face_id):
        assert face_id in self.faces
        return torch.load(f'{face_id}.pt')

    def writeFace(self, face_vector, face_id):
        torch.save(face_vector, f'{face_id}.pth')
        self.updateFaces()

    def deleteFace(self, face_id):
        assert face_id in self.faces
        os.remove(f'{face_id}.pt')
        self.updateFaces()

    def updateFaces(self):
        self.faces = os.listdir(self.face_dir)

    def runRecognition(self, frame, face_id):
        output = self.model(frame)
        face_vector = self.loadFace(face_id)
        return self.withinThreshold(output, face_vector)

    def withinThreshold(self, face_vector1, face_vector2, threshold=1):
        return True if torch.mean(f.pairwise_distance(face_vector1, face_vector2)) < threshold else False

    def preprocessFrame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            w = max(w, h)
            h = w #SQUARE-IFY
            image = frame[y:y+h, x:x+w] #CROP IMAGE
            image = cv2.resize(image, (self.input_width, self.input_height))
            image = Image.fromarray(image) 
            image = self.preprocess(image).unsqueeze(0) #(1, 3, 224, 224)

            return True, frame, image

        return False, frame, None

    def loadModelWeights(self, weight_fn):
        assert os.path.isfile(weight_fn), "{} is not a file.".format(weight_fn)
        state = torch.load(weight_fn)
        weight = state['weight']
        it = state['iterations']
        self.model.load_state_dict(weight)
        print("Checkpoint is loaded at {} | Iterations: {}".format(weight_fn, it))

