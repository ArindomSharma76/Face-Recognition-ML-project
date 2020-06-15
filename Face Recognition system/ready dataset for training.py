
import cv2
import numpy as np#not used in the program
#we need harcascade classifier to make the machine know that what it is seeing is actually a face of a human
#classifiers classify the objects that defines face, hair, cheek etc.
face_classifier = cv2.CascadeClassifier('C:/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

#extracting face features
def face_extractor(img):
#we have the image in RGB but converting it into grayscale as it is easy to use
     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     #detectMultiScale Detects objects of different sizes in the input image. The detected objects are returned as a list . of rectangles. .
     faces=face_classifier.detectMultiScale(gray,1.3,5) #1.3 is the scaling factor and min neighbours is 5, higher no. of neighbours is for more accuracy
     if faces is():    #if face is not there
         return None

     for(x,y,w,h) in faces:#x for coloums and y for rows, w is width and h is height
         cropped_face=img[y:y+h, x:x+w]
     return cropped_face

#configuring camera
cap=cv2.VideoCapture(0)
count=0
while True:
    ret,frame= cap.read()
    if face_extractor(frame) is not None:  #if a face is detected in front of webcam
        count+=1
        #camera frame size needs to be similar to our face size
        face=cv2.resize(face_extractor(frame),(200,200))#200x200 dimension of the frame required
        face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)#resize face is converted to gray scale
        #saving face's values in a address
        file_name_path='C:/Users/Arindom/Desktop/My projects/Face Recognition system/faces/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)
        #to count no. of images
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)#(50,50) is the starting point,cv2.FONT_HERSHEY_COMPLEX is the font style,1 is the scaling of font,(0,255,0) is the color and 2 is the font thickness
        cv2.imshow('Face Cropper',face)
    else:
        print('Face not found')
        pass
    if cv2.waitKey(1)==13 or count==100:# the program will close either we press enter or after it takes 100 samples
        break
cap.release()#to close the camera
cv2.destroyAllWindows()
print('Collecting samples complete!!!!')
