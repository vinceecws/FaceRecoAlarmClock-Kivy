import os
import torch
import torch.nn.functional as f
import cv2
import uuid
from PIL import Image
from torchvision import transforms
from Siamese_MobileNetV2.src.models.Siamese_MobileNetV2 import Siamese_MobileNetV2_Triplet, TripletLoss

class FaceRecognitionAPI():
    def __init__(self, face_dir, weight_dir, haar_dir):
        assert os.path.isdir(face_dir), "{} is not a folder.".format(face_dir)

        self.face_dir = face_dir
        self.weight_dir = weight_dir
        self.haar_dir = haar_dir

        #Initialize faces
        self.faces = os.listdir(face_dir)

        #Initialize Haar
        self.face_cascade = cv2.CascadeClassifier(haar_dir)

        #Initialize model
        self.input_width = 224
        self.input_height = 224
        self.model = Siamese_MobileNetV2_Triplet()
        self.model.eval()
        self.loadModelWeights(weight_dir)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def processAndRegister(self, images, current_id=None):
        vectors = []
        for image in images:
            valid, _, input_image = self.preprocessFrame(image)

            if valid:
                vectors.append(self.computeVector(input_image))

        face_vector = torch.cat(vectors, axis=0)
        self.registerFace(face_vector, current_id)

    def registerFace(self, face_vector, current_id=None):
        if not current_id:
            current_id = str(uuid.uuid4()) #Generate new id

        self.writeFace(face_vector, current_id) #Add new entry

    def updateFaces(self):
        self.faces = os.listdir(self.face_dir)

    def runRecognition(self, frame, face_id):
        output = self.model(frame)
        face_vector = self.loadFace(face_id)
        return self.withinThreshold(output, face_vector)

    def computeVector(self, image):
        return self.model(image)

    def withinThreshold(self, face_vector1, face_vector2, threshold=0.1):
        score = torch.mean(TripletLoss.dist(face_vector1, face_vector2))
        print(score)
        return True if score < threshold else False

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

    '''
        I/O Utilities
    '''
    def loadModelWeights(self, weight_fn):
        assert os.path.isfile(weight_fn), "{} is not a file.".format(weight_fn)
        state = torch.load(weight_fn)
        weight = state['weight']
        it = state['iterations']
        self.model.load_state_dict(weight)
        print("Checkpoint is loaded at {} | Iterations: {}".format(weight_fn, it))

    def loadFace(self, face_id):
        face_id = f'{face_id}.pth'
        assert face_id in self.faces
        return torch.load(os.path.join(self.face_dir, face_id))

    def writeFace(self, face_vector, face_id):
        face_id = f'{face_id}.pth'
        torch.save(face_vector, os.path.join(self.face_dir, face_id))
        self.updateFaces()

    def deleteFace(self, face_id):
        face_id = f'{face_id}.pth'
        assert face_id in self.faces
        os.remove(os.path.join(self.face_dir, face_id))
        self.updateFaces()

