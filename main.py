import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'ImagesAttendance'
images = [] #creates a list of all images to import

classNames = [] #take names from images automatically
myList = os.listdir(path) #grab list of names of images in this folder
print(myList)

#use these names from myList and import their respective images
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) #removes .jpg at the end
print(classNames)

#function that will compute all our encodings for us
def findEncodings(images):
    encodeList = []
    for img in images:
        imgElon = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0] #finds the encodings in the image
        encodeList.append(encode) #appends encodings to encoded list
    return encodeList

#function that marks attendance with name and time arrived
def markAttendace(name):
    with open('attendance.csv', 'r+') as f: #read and write at the same time
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeListKnown = findEncodings(images)
print('Encoding Complete')

#initialize webcam to take image that will match to the encodings
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) #resize image to 1/4 the size
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert image to rgb to acount for light changes in an image

    facesCurFrame = face_recognition.face_locations(imgS)  #locate the faces in current frame
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) #finds the encodings in the image

    #finding the matches by iterating through all the faces found in current frame
    #compare faces with all the encodings we found before

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame): #we use zip because we want them in the same loop
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) #compare known encodings to new encode face
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) #find face distances between these two encodings

        matchIndex = np.argmin(faceDis) #finds smallest value in faceDis

        #Prints the name of the found face in matchIndex from class names
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)

            #draw triangle on image
            y1, x2, y2, x1 = faceLoc

            #y1, x2, y2, x1 = y1*4, x2*5, y2*5, x1*5 #multiply our image by 4 since we scalled down our image to 1/4 the size
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2) #put text on image
            markAttendace(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)



#faceLocTest = face_recognition.face_locations(imgTest)[0] #locate the face
#encodeTest = face_recognition.face_encodings(imgTest)[0] #find the encodings on the located face
#cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)#draw a rectangle where the face location is

#results = face_recognition.compare_faces([encodeElon], encodeTest) #compares both images and predicts if it matches or not
