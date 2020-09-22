import numpy as np
import cv2
import glob
import os


if not os.path.exists('Detected_Iris'):
    os.makedirs('Detected_Iris')
    
passed = 0
failed = 0
for jj in glob.glob('./UBIRIS_200_150/Sessao_2/**/*.jpg'):
    img = cv2.imread(jj,0)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    thresh4 = cv2.medianBlur(thresh4,13)
    thresh4 = cv2.Canny(thresh4,100,180)
    circles = cv2.HoughCircles(thresh4,cv2.HOUGH_GRADIENT,1,100,param1=60,param2=15,minRadius=35,maxRadius=70)
    
    if type(circles) == np.ndarray:
        circles = np.uint16(np.around(circles))
        print(jj,circles,file = open('centers.txt','a'))
        for i in circles[0,:]:
            cv2.circle(img,(i[0],i[1]),i[2],(0,0,255),2)
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
            passed = passed + 1
        
    else:
        failed = failed + 1
    
    cv2.imwrite('./Detected_Iris/' + jj.split('\\')[-1], img)
    print(passed,failed)
