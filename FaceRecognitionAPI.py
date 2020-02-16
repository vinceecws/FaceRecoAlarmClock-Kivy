import os
import torch
import torch.nn.functional as f
import cv2
import uuid
import pickle
from PIL import Image
from torchvision import transforms
from Siamese_MobileNetV2.src.models.Siamese_MobileNetV2 import Siamese_MobileNetV2_Triplet, TripletLoss
from MobileFaceNet_Pytorch.core import model as MobileFaceNetModel

class FaceRecognitionAPI():
    def __init__(self, face_dir, weight_dir, haar_dir):
        assert os.path.isdir(face_dir), "{} is not a folder.".format(face_dir)

        self.face_dir = face_dir
        self.weight_dir = weight_dir
        self.haar_dir = haar_dir

        #Initialize faces
        self.syncFaces()
        self.loadCurrent()
        print(f'Current face id: {self.current_face_id}')

        #Initialize Haar
        self.face_cascade = cv2.CascadeClassifier(haar_dir)

        #Initialize model
        self.input_width = 224
        self.input_height = 224
        #self.model = Siamese_MobileNetV2_Triplet()
        #self.model.eval()
        self.model = MobileFaceNetModel.MobileFacenet()
        self.loadModelWeights(weight_dir, key='net_state_dict')
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
        current_id = self.registerFace(face_vector, current_id)
        return current_id

    def batchRegister(self, images, current_id=None):
        vectors = []
        for image in images:
            vectors.append(self.computeVector(image))
            face_vector = torch.cat(vectors, axis=0)
            current_id = self.registerFace(face_vector, current_id)

        return current_id

    def registerFace(self, face_vector, current_id=None):
        if current_id is None:
            current_id = str(uuid.uuid4()) #Generate new id

        self.writeFace(face_vector, current_id) #Add new entry
        return current_id

    def syncFaces(self):
        self.faces = [face for face in os.listdir(self.face_dir) if face != 'current']

    def runRecognition(self, frame):
        assert self.current_face is not None and self.current_face_id is not None, 'Set up a current face vector first, call loadCurrent()'
        output = self.model(frame)
        return self.withinThreshold(output, self.current_face)

    def computeVector(self, image):
        return self.model(image)

    def withinThreshold(self, face_vector1, face_vector2, threshold=2700, metric='min'):
        assert metric in ['mean', 'min']
        if metric == 'mean':
            score = torch.mean(TripletLoss.dist(face_vector1, face_vector2))
        elif metric == 'min':
            score = torch.min(TripletLoss.dist(face_vector1, face_vector2))
        return True if score < threshold else False

    def preprocessFrame(self, frame):
        image = cv2.resize(frame, (self.input_width, self.input_height))
        image = Image.fromarray(image) 
        image = self.preprocess(image).unsqueeze(0) #(1, 3, 224, 224)

        return True, frame, image

    def cropAndPreprocessFrame(self, frame):
        original = frame
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

            return True, frame, image, original

        return False, frame, None, original

    '''
        I/O Utilities
    '''
    def loadModelWeights(self, weight_fn, key='weight'):
        assert os.path.isfile(weight_fn), "{} is not a file.".format(weight_fn)
        state = torch.load(weight_fn, map_location=torch.device('cpu'))
        weight = state[key]
        if 'iterations' in state.keys():
            it = state['iterations']
            print("Checkpoint is loaded at {} | Iterations: {}".format(weight_fn, it))
        else:
            print("Checkpoint is loaded at {}".format(weight_fn))

        self.model.load_state_dict(weight)

    def loadFace(self, face_id):
        face_id = f'{face_id}.pth'
        assert face_id in self.faces
        return torch.load(os.path.join(self.face_dir, face_id))

    def writeFace(self, face_vector, face_id):
        face_id = f'{face_id}.pth'
        torch.save(face_vector, os.path.join(self.face_dir, face_id))
        self.syncFaces()

    def deleteFace(self, face_id):
        face_id = f'{face_id}.pth'
        assert face_id in self.faces
        os.remove(os.path.join(self.face_dir, face_id))
        self.syncFaces()

    def loadCurrent(self):
        current_dir = os.path.join(self.face_dir, 'current')
        assert os.path.isfile(current_dir), f'Path to \'{current_dir}\' does not exist'
        with open(current_dir, "rb") as file: 
            self.current_face_id = pickle.load(file)

        if self.current_face_id:
            self.current_face = self.loadFace(self.current_face_id)
            return True

        self.current_face = None
        return False

    def saveCurrent(self, face_id):
        current_dir = os.path.join(self.face_dir, 'current')
        self.current_face_id = face_id
        self.current_face = self.loadFace(self.current_face_id)
        with open(current_dir, "wb") as file:
            pickle.dump(self.current_face_id, file)
        print('Current face id: {}'.format(self.current_face_id))

