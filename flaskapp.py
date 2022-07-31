

from flask import Flask, render_template, Response
import cv2
from sklearn import utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import os
import torch
import sys
sys.path.append('../')
from utils.logger import Logger
from torchvision import transforms
import numpy as np
from PIL import Image


app=Flask(__name__)

class VggFeatures(nn.Module):
    def __init__(self, drop=0.2):
        super().__init__()

        self.conv1a = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(64, out_channels=64, kernel_size=3, padding=1)

        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4a = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv2d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1a = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)

        self.bn2a = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(128)

        self.bn3a = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)

        self.bn4a = nn.BatchNorm2d(512)
        self.bn4b = nn.BatchNorm2d(512)

        self.lin1 = nn.Linear(512 * 2 * 2, 4096)
        self.lin2 = nn.Linear(4096, 4096)

        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)

        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool(x)
        # print(x.shape)

        x = x.view(-1, 512 * 2 * 2)
        x = F.relu(self.drop(self.lin1(x)))
        x = F.relu(self.drop(self.lin2(x)))

        return x


class Vgg(VggFeatures):
    def __init__(self, drop=0.2):
        super().__init__(drop)
        self.lin3 = nn.Linear(4096, 7)

    def forward(self, x):
        x = super().forward(x)
        x = self.lin3(x)
        return x


''''''



def video_capture(net):
    face_cascade = cv2.CascadeClassifier('C:\Python\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

    mu,st = 0,255
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(mu,), std=(st,))
        ])

    lb = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
    def check(gray_face):
        img = cv2.resize(gray_face, (40,40)).astype(np.float64)
        img = Image.fromarray(img)
        img = test_transform(img)
        img.unsqueeze_(0)
        outputs = net(img)
        _, preds = torch.max(outputs.data, 1)
        return int(preds.data[0])

    cap = cv2.VideoCapture(0)
    while True:
            # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            face = gray[y:y+h,x:x+w]
            a = check(face)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            img = cv2.putText(img, lb[a], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 1, cv2.LINE_AA)
        # Display
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



    
@app.route('/')
def index():
    #return render_template('index.html')
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    net = Vgg()
    #net = net.eval()
    epoch = 200
    path = os.path.join('demo','cp_demo', 'epoch_' + str(epoch))
    print("Systhpath: ",path)
    logger = Logger()
    checkpoint = torch.load(path, map_location=torch.device('cpu'))

    logger.restore_logs(checkpoint['logs'])
    net.load_state_dict(checkpoint['params'])
    print("Network Restored!")
    return Response(video_capture(net), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)