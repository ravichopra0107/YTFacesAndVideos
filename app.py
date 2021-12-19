import numpy as np
import cv2
import pickle
from vidgear.gears import CamGear
import imutils
import sys
from youtube_search import YoutubeSearch
import time

# Constant
font = cv2.FONT_HERSHEY_SIMPLEX
textColor = (255, 255, 255)
stroke = 2
rectColor = (255, 0, 0)


# Load model
face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Load labels
labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# Cap init
cap = cv2.VideoCapture(0)
tempID = -1


def getStream(name):
    results = YoutubeSearch(name, max_results=1).to_dict()
    id = results[0]['id']
    options = {"CAP_PROP_FPS": 10}
    # Initialise stream
    stream = CamGear(source=f'https://youtu.be/{id}', stream_mode=True,
                     logging=True, **options).start()
    return stream


stream = ""
while True:
    cv2.namedWindow('frame')
    ret, frame = cap.read()
    # For recognizer
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45:
            name = labels[id_]
            cv2.putText(frame, name, (x, y), font, 1,
                        textColor, stroke, cv2.LINE_AA)
            if tempID != id_:
                tempID = id_
                stream = getStream(name)
            # Read stream
            videoFrame = stream.read()
            # Stream End
            if videoFrame is None:
                sys.exit()
            videoFrame = imutils.resize(videoFrame, width=w+100)
            videoFrame_h = videoFrame.shape[0]
            videoFrame_w = videoFrame.shape[1]
            try:
                frame[y:y+videoFrame_h,
                      x:x+videoFrame_w] = videoFrame
            except:
                continue

            def on_mouse(event, x_touch, y_touch, flags, params):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if(y_touch >= y and y_touch <= y+videoFrame_h and x_touch >= x and x_touch <= x+videoFrame_w):
                        startTime = time.time()
                        while True:
                            f = stream.read()
                            if f is None:
                                break
                            f = imutils.resize(f, width=1200)
                            nowTime = time.time()
                            fpsLimit = 1/24
                            if ((nowTime - startTime)) > fpsLimit:
                                cv2.imshow("f", f)
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord("q"):
                                    cv2.destroyAllWindows()
                                    stream.stop()
                                    break
                                startTime = time.time()

        cv2.setMouseCallback('frame', on_mouse)

    cv2.imshow("frame", frame)

# Draw rectangle
# x_end = x+w
# y_end = y+h
# cv2.rectangle(frame, (x, y), (x_end, y_end), rectColor, stroke)
    # cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
