import re
from unittest import result
import numpy as np
import cv2
import streamlit as st

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode


from sklearn import utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import torch
import sys
from utils.logger import Logger
from torchvision import transforms
import numpy as np
from PIL import Image,ImageEnhance
import av

from models.vgg import Vgg

# load model
net = Vgg()
#net = net.eval()
epoch = 200
path = os.path.join('demo','cp_demo', 'epoch_' + str(epoch))
print("Systh: ",path)
checkpoint = torch.load(path, map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['params'])
print("Network Restored!")

#load face
try:
    face_cascade = cv2.CascadeClassifier('C:\Python\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

#########
def Ferprocess(upload_img):
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

    #cap = cv2.VideoCapture(0)
    #img = frame.to_ndarray(format="bgr24")

    #image gray
    gray = cv2.cvtColor(upload_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face = gray[y:y+h,x:x+w]
        a = check(face)
        cv2.rectangle(upload_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        upload_img = cv2.putText(upload_img, lb[a], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
        
    return upload_img

##########

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame)-> av.VideoFrame:
        self.img = frame.to_ndarray(format="bgr24")
        self.output = Ferprocess(self.img)
        return self.output

def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    activiteis = ["Home", "Webcam Face Detection","Upload Image", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Nguyen Thanh Dam    
            Email : damntse150556@fpt.edu.vn
            [Github] (https://github.com/DamNT055/Facial-expression-recognition-Vgg-)""")
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has three functionalities.

                 1. Real time face detection using web cam feed.

                 2. Real time face emotion recognization.
                 
                 3. Detect emotion from image uploaded.


                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        media_stream_constraints={"video": True, "audio": False},
                        video_processor_factory=Faceemotion)

    elif choice == "Upload Image":
        st.subheader("Face Detection")
        img_file=st.file_uploader("Upload File",type=['png','jpg','jpeg'])
        if img_file is not None:
            up_image=Image.open(img_file)
            st.image(up_image)
        enhance_type=st.sidebar.radio("Enhance type",["Originial","Gray-scale","Contrast","Brightness","Blurring"])
        if enhance_type=="Gray-scale":
            new_img=np.array(up_image.convert('RGB'))
            img=cv2.cvtColor(new_img,1)
            gray=cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            st.image(gray)
        if enhance_type=="Contrast":
            c_make=st.sidebar.slider("Contrast",0.5,3.5)
            enhacer=ImageEnhance.Contrast(up_image)
            img_out=enhacer.enhance(c_make)
            st.image(img_out)
        if enhance_type=="Brightness":
            b_make=st.sidebar.slider("Brightness",0.5,3.5)
            enhacer=ImageEnhance.Brightness(up_image)
            img_bg=enhacer.enhance(b_make)
            st.image(img_bg)
        if enhance_type=="Blurring":
            br_make=st.sidebar.slider("Blurring",0.5,3.5)
            br_img=np.array(up_image.convert('RGB'))
            b_img=cv2.cvtColor(br_img,1)
            blur=cv2.GaussianBlur(b_img,(11,11),br_make)
            st.image(blur)      
        task=["Faces","Eye","Face Emotion Detection"]
        feature_choice=st.sidebar.selectbox("Find Feature",task)
        if st.button("Process"):
            if feature_choice=="Faces":
                result_img,result_faces=detect_faces(up_image)
                st.image(result_img)
                st.success("Found {} faces".format(len(result_faces)))
            if feature_choice=="Eye":
                result_img,result_faces=detect_eye(up_image)
                st.image(result_img)
                st.success("Found {} Eyes".format(len(result_faces)))  
            if feature_choice=="Face Emotion Detection":
                new_img=np.array(up_image.convert('RGB'))
                result_img = Ferprocess(new_img)
                st.image(result_img)
    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by AIP391_ Team 5 using Streamlit Framework, Opencv, Pytorch library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. If you have any suggestion or wnat to comment just write a mail at damntse150556@fpt.edu.vn. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


#Code detect face and eye


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes=cv2.CascadeClassifier('haarcascade_eye.xml')
def detect_faces(up_image):
	detect_img=np.array(up_image.convert('RGB'))
	new_img1=cv2.cvtColor(detect_img,1)
	# gray=cv2.cvtColor(new_img1,cv2.COLOR_BGR2GRAY)
	faces=faceDetect.detectMultiScale(new_img1,1.3,5)
	for x,y,w,h in faces:
		cv2.rectangle(new_img1,(x,y),(x+w,y+h),(255,255,0),2)
	return new_img1,faces
def detect_eye(up_image):
	detect_img=np.array(up_image.convert('RGB'))
	new_img1=cv2.cvtColor(detect_img,1)
	# gray=cv2.cvtColor(new_img1,cv2.COLOR_BGR2GRAY)
	faces=eyes.detectMultiScale(new_img1,1.3,5)
	for x,y,w,h in faces:
		cv2.rectangle(new_img1,(x,y),(x+w,y+h),(255,255,0),2)
	return new_img1,faces

""""""

if __name__ == "__main__":
    main()
