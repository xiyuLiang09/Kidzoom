# -*- coding: utf-8 -*-


import numpy as np
import cv2

step=10
global points
def onMouse(event, x, y, flags, param):
    global points #声明引用全局变量    
    if event == cv2.EVENT_LBUTTONDOWN: #鼠标点击弹起事件 
        print(x,y)
        r=100000000
        pi=5
        for i in range(len(points)):            
            if r>((x-points[i][0])**2+(y-points[i][1])**2):
                r=(x-points[i][0])**2+(y-points[i][1])**2
                pi=i
        print("point:",pi)
        if pi!=5:            
            points[pi]=[x,y] 
            print(points)
    
def drawRec(img,points,flag):
    """
    画出境界区域
    """
    if flag:
        color=(255,0,255)
    else:
        color=(0,255,0)
        
    for i in range(len(points)):        
        cv2.circle(img, (points[i][0],points[i][1]), 10, color, 2) 
        if i!=3:
            cv2.line(img, (points[i][0],points[i][1]),(points[i+1][0],points[i+1][1]), color, 4)
        else:
            cv2.line(img, (points[i][0],points[i][1]),(points[0][0],points[0][1]),color, 4)
    return img    
def isinRec(point,points):
    """
    是否在边框内
    """

    [x1,y1] = points[0]
    [x2,y2] = points[1]
    [x3,y3] = points[2]
    [x4,y4] = points[3]
    p1 = (x1-x4)*(point[1]-y4)-(y1-y4)*(point[0]-x4)
    p2 = (x2-x1)*(point[1]-y1)-(y2-y1)*(point[0]-x1)
    p3 = (x3-x2)*(point[1]-y2)-(y3-y2)*(point[0]-x2)
    p4 = (x4-x3)*(point[1]-y3)-(y4-y3)*(point[0]-x3)
    #print(a,b,c,d)
    if (p1>0 and p2>0 and p3>0 and p4>0) or (p1<0 and p2<0 and p3<0 and p4<0):
        return True
    else:
        return False


if __name__ == '__main__':    
    cam = cv2.VideoCapture(r"./inference/images/QQ视频20230630231142.mp4")

    # 检查视频是否成功打开
    if not cam.isOpened():
        print("Error: Cannot open video file.")
        exit()

    ret, prev = cam.read()

    # 检查视频帧是否读取成功
    if not ret:
        print("Error: Cannot read video frame.")
        exit()

    # ret, prev = cam.read()
    H=prev.shape[0]
    W=prev.shape[1]
    # 初始化矩形框
    points=[
            [50,H-50],
            [50,50],
            [W-50,50],        
            [W-50,H-50]                               
            ]    
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('warning') #创建窗口    
    cv2.setMouseCallback('warning',onMouse) #设置窗口鼠标回调函数

    while True:
        ret, frame = cam.read()
        if ret:            
            H=frame.shape[0]
            W=frame.shape[1]
        else:
            H=100
            W=100

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        """
        prevImg – First 8-bit single-channel input image.   输入单通道图片
        nextImg – Second input image of the same size and the same type asprevImg . 下一帧图片。
        flow – Computed flow image that has the same size as prevImg and typeCV_32FC2 .输出的双通道flow
        pyrScale – Parameter specifying the image scale (<1) to build pyramids for each image. pyrScale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.金字塔上上下两层之间的尺度关系。
        levels – Number of pyramid layers including the initial image.levels=1 means that no extra layers are created and only the original images are used. 金字塔层数
        winsize – Averaging window size. Larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.均值窗口大小，越大越能denoise并且能够检测快速移动目标，但是会引起模糊运动区域。
        iterations – Number of iterations the algorithm does at each pyramid level.迭代次数。
        polyN – Size of the pixel neighborhood used to find polynomial expansion in each pixel. Larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field. Typically, polyN =5 or 7.
        polySigma – Standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion. ForpolyN=5 , you can set polySigma=1.1 . For polyN=7 , a good value would be polySigma=1.5
        flag-OPTFLOW_USE_INITIAL_FLOW  OPTFLOW_FARNEBACK_GAUSSIAN
        """
        prevgray = gray
        # 绘制线
        h, w = gray.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)#等差数列
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)

        line = []

        movepoint=0        
        
        for l in lines:
            if l[0][0]-l[1][0]>3 or l[0][1]-l[1][1]>3:                
                line.append(l)
                if isinRec(l[0].tolist(),points):
                    movepoint+=1

        if movepoint > 10:  
            # print(movepoint + '注意！儿童进入危险区！')
            print("注意！儿童进入危险区！")
            cv2.polylines(frame, line, 0, (255,255,0))
            pimg=drawRec(frame,points,True)  
        else:
            print('安全')
            cv2.polylines(frame, line, 0, (0,255,0))
            pimg=drawRec(frame,points,False)              
              
        cv2.imshow('warning', pimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()       
 
