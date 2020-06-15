import cv2
import numpy as np #for any mathematical operation
from os import listdir    # os is inbuilt  lib and listdir is a class of module os. It is used to fetch data from directory
from os.path import isfile, join

#first we need path where images are stored
data_path='C:/Users/Arindom/Desktop/My projects/Face Recognition system/faces/'
#new way of writing for loop and giving condition in a single line
onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]#now we need data that is in this location and all data comes in form of list
Training_Data, Lables=[], [] #here one list will contain the training data i.e the images of my face and other list will contain labels

#now calling the datasets
for i, files in enumerate(onlyfiles):#enumerate provides iterations upto the number of files in onlyfiles variables
    image_path=data_path+onlyfiles[i]#it will provide index of the images
    images=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)#it will read all the grayscale images
    Training_Data.append(np.asarray(images, dtype=np.uint8))#append training images data in the form of an array
    Lables.append(i)#lables will append all i values

Lables=np.asarray(Lables, dtype=np.int32)#Calling the labels

#Now building the model
model = cv2.face.LBPHFaceRecognizer_create()#This is the classifier used for this we have to pip install opencv-contrib-python package

model.train(np.asarray(Training_Data), np.asarray(Lables))
print("Model Training Completed!!!!")

#we need the face classifier
face_classifier = cv2.CascadeClassifier('C:/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

#creating a face for face detection
def face_detector(img, size=0.5):#img is the image from camera frame
     gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#converting to gray scale
     faces=face_classifier.detectMultiScale(gray, 1.3, 5)#1.3 is the scaling factor and min neighbours is 5, higher no. ofr neighbours is for more accuracy
     if faces is():
         return img,[]#if there is no image it will return empty list
     for(x,y,w,h) in faces:#contains data of image's region of interest
         cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),4)#create a rectangle around the image,here 2 is thickness of rectangle boundary
         roi=img[y:y+h, x:x+w]#region of interest
         roi=cv2.resize(roi, (200,200))
     return img,roi
cap=cv2.VideoCapture(0)#camera is on
while True:
    ret, frame=cap.read()#frame starts reading i.e we start getting the video stream

    image, face=face_detector(frame)#from we are getting face and image from face_detector
    try:
        face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)#converting the received face to gray
        result=model.predict(face)#predicting my face 

        if result[1]<500: #500 is a pseudo value we can take any other value
            confidence=int(100*(1-(result[1])/300))#calculating confidence value, we are sub from 1 bcz in python value difference remains atleast 1
            #confidence will give upto what percent face is matching
            display_string=str(confidence)+'% confidence it is user'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(25,12,255))#this is the features of the confidence% string



        if confidence>80:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
            cv2.imshow('Face Cropper',image)
        else:
            cv2.putText(image, "locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            cv2.imshow('Face Cropper', image)
    except:#if there is no face
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

        cv2.imshow('Face Cropper', image)
        pass
    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()