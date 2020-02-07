#Taken originally from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html 
#and modified for testing

import numpy as np
import cv2

def horizontalSpeedLimit(flow, limit=6.0):
    return np.max(flow[:,:,0]) > limit

def verticalSpeedLimit(flow, limit=6.0):
    return np.max(flow[:,:,1]) > limit

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240);

ret, frame1 = cap.read()
frame1 = cv2.flip(frame1, 1)
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

while(1):
    ret, frame2 = cap.read()
    frame2 = cv2.flip(frame2, 1)
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    if horizontalSpeedLimit(flow) or verticalSpeedLimit(flow):
        print('Too fast')

    cv2.imshow('frame2',frame2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    prvs = next

cap.release()
cv2.destroyAllWindows()

