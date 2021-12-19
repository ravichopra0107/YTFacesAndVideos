import numpy as np
import cv2
import  pickle
from vidgear.gears import CamGear
from youtube_search import YoutubeSearch
import imutils
face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
labels={}
with open("labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
cap=cv2.VideoCapture(0)



while(True):
    ret,frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]

        id_,conf=recognizer.predict(roi_gray)
        if conf>=45:
            results = YoutubeSearch(labels[id_], max_results=1).to_dict()
            video_id=results[0]['id']
            stream = CamGear(source=f'https://youtu.be/{video_id}', stream_mode=True).start()
            while True:
              cv2.namedWindow('frame2')

              frame2 = stream.read()
              if frame is None:
                  break
              frame2 = imutils.resize(frame, width=w + 100)
              frame2_h = frame2.shape[0]
              frame2_w = frame2.shape[1]
              frame[y:y + frame2.shape[0],
              x:x+ frame2.shape[1]] = frame2
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

            img_item="myimg.png"
            cv2.imwrite(img_item,roi_gray)

            color=(255,0,0)
            stroke=2
            x_end=x+w
            y_end=y+h
            cv2.rectangle(frame,(x,y),(x_end,y_end),color,stroke)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF==ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
stream.stop()