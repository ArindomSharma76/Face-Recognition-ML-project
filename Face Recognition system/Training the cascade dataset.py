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



