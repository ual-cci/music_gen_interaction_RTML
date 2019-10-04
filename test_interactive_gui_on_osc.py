# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_trackbar/py_trackbar.html#trackbar
# https://github.com/kivy/oscpy

import cv2
import numpy as np
from oscpy.client import OSCClient

address = "127.0.0.1"
port = 8000
osc = OSCClient(address, port)

def onChange(x):
    print("Sending x=",x)
    osc.send_message(b'/send_i', [x])

# Create a black image, a window
h = 75
img = np.zeros((h,512,3), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
position = (0,h-25)
fontScale = 1
fontColor = (128,128,128)
lineType = 2


cv2.namedWindow('InteractiveMusicGeneration')

# create trackbars for color change
cv2.createTrackbar('Percentage','InteractiveMusicGeneration',0,1000,onChange)



while(1):
    #also keep another inf. loop

    cv2.imshow('InteractiveMusicGeneration',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('Percentage','InteractiveMusicGeneration')
    r = int(r)

    img[:] = [r,r,r]
    cv2.putText(img, 'Select value: (0 to 1000 => %)',position,font,fontScale,fontColor,lineType)

cv2.destroyAllWindows()
