from cv2 import cv2
import os
import imutils

user = 'Jeffrey Sanchez'
directory = './images'
userDirectory = f"{directory}/{user}"

if not os.path.exists(userDirectory):
    print(f"New folder created: {userDirectory}")
    os.makedirs(userDirectory)

count = 0
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    img = imutils.resize(img, width=640)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frameCopy = img.copy()

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = frameCopy[y:y+h, x:x+w]
        face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(userDirectory + '/face_{}.jpg'.format(count), face)
        count += 1
    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()
