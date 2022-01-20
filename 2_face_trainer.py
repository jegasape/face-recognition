import cv2
import os
import numpy as np

directory = './images'
people = os.listdir(directory)
print(f'People: {people}')

labels = []
facesData = []
label = 0

for nameDir in people:
    personPath = f"{directory}/{nameDir}"
    print('Reading....')

    for fileName in os.listdir(personPath):
        print('Faces: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName, 0))
    label = label + 1
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Training...")
face_recognizer.train(facesData, np.array(labels))
face_recognizer.write('modelLBPHFace.xml')
print("Model saved...")
