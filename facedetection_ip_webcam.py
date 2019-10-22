# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 22:31:07 2019

@author: anantha.n.srinivasan
"""

import requests
import cv2
import numpy as np


url="http://192.168.0.100:8080/shot.jpg"

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")

while True:
    img_resp=requests.get(url)
    img_array=np.array(bytearray(img_resp.content),dtype=np.uint8)
    img=cv2.imdecode(img_array,-1)
    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=img[y:y+h, x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('cam_show',img)        
    #cv2.waitKey(0)
    k=cv2.waitKey(30)
    if k==27:
        break
    
cv2.destroyAllWindows()
    
    