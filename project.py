import cv2 as cv
import pafy
import numpy as np

win_title = "window"
mouse_is_pressing = False
start_x, start_y, end_x, end_y = 0,0,0,0
step = 0

def swap(v1, v2):
    temp = v1
    v1 = v2
    v2 = temp

def mouse_callback(event, x, y, flags, param):

    global step, start_x, end_x, start_y, end_y, mouse_is_pressing

    if event == cv.EVENT_LBUTTONDOWN:
        step = 1

        mouse_is_pressing = True
        start_x = x
        start_y = y

    elif event == cv.EVENT_MOUSEMOVE:
        if mouse_is_pressing:

            end_x = x
            end_y = y

            step = 2

    elif event == cv.EVENT_LBUTTONUP:
        mouse_is_pressing = False

        end_x = x
        end_y = y
        
        step = 3

url = 'https://www.youtube.com/watch?v=tZixREYOIZQ'
video = pafy.new(url)
best = video.getbest()

cap=cv.VideoCapture(best.url)

Width = 500
Height = 300

cap.set(cv.CAP_PROP_FRAME_WIDTH,Width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,Height)
rows,cols=Height,Width


cv.namedWindow(win_title)
cv.setMouseCallback(win_title, mouse_callback)

while True:

    ret, frame = cap.read()
    if ret == False:
        break
     
    if step == 1:
        cv.circle(frame, (start_x, start_y), 
            10, (0, 255, 0), -1)

    elif step == 2:
        cv.rectangle(frame, (start_x, start_y), (end_x, end_y), 
            (0, 255, 0), 3)

    elif step == 3:
        if start_x > end_x:
            swap(start_x, end_x)
            swap(start_y, end_y)

        w=end_x-start_x
        h=end_y-start_y
        rate = 15
        
        if w and h:
            roi = frame[start_y:start_y+h,start_x:start_x+w]

            roi = cv.resize(roi,(w//rate,h//rate))
            roi = cv.resize(roi,(w,h),interpolation=cv.INTER_AREA)
            
            frame[start_y:start_y+h,start_x:start_x+w]=roi
                    
    cv.imshow(win_title, frame)

    if cv.waitKey(25) > 0:
        break; 