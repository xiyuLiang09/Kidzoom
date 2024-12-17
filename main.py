import argparse
import numpy as np
from pathlib import Path
import random
from typing import List, Tuple
from sympy import plot
import torch
import cv2
from keypoint import get_keypoints
from models.experimental import attempt_load
from utils.datasets import LoadImages, letterbox

from utils.general import check_img_size, increment_path, non_max_suppression, scale_coords
from utils.plots import plot_skeleton_kpts

# 全局变量
step = 10
window_points: List[Tuple]

def onMouse(event, x, y, flags, params):
    global window_points #声明引用全局变量    
    # print(f"event: {event}")
    # print(f"flags: {flags}")
    # print(f"params: {params}")
    # if event == cv2.EVENT_LBUTTONDOWN and window_points != None: #鼠标点击弹起事件 
    if flags == cv2.EVENT_FLAG_LBUTTON and window_points != None: #鼠标点击弹起事件 
        # print(x,y)
        r=100000000
        pi=5
        for i in range(len(window_points)):            
            if r>((x-window_points[i][0])**2+(y-window_points[i][1])**2):
                r=(x-window_points[i][0])**2+(y-window_points[i][1])**2
                pi=i
        print("point:",pi)
        if pi!=5:            
            window_points[pi]=[x,y] 
            print(window_points)


def drawRec(img,points,flag):
    """
    画出境界区域
    """
    thickness = 2
    color = (255, 0, 255) if flag else (0, 255, 0)
        
    for i in range(len(points)):        
        cv2.circle(img, (points[i][0],points[i][1]), 10, color, thickness=thickness)
        if i!=3:
            cv2.line(img, (points[i][0],points[i][1]),(points[i+1][0],points[i+1][1]), color, thickness=thickness)
        else:
            cv2.line(img, (points[i][0],points[i][1]),(points[0][0],points[0][1]),color, thickness=thickness)
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


def init_window_points(img0):
    global window_points
    img = letterbox(img0)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    weights = "./window.pt"
    imgsz = 640

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    half = (device.type == "cuda")
    
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Warmup
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for _ in range(3):
            model(img, augment=opt.augment)[0]

    with torch.no_grad():
        pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    if len(pred):
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # # Print results
        # for c in det[:, -1].unique():
        #     n = (det[:, -1] == c).sum()  # detections per class
        #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Init window points
        for *xyxy, _, _ in reversed(det):
            window_points = [
                (int(xyxy[2]), int(xyxy[1])),   # 左下
                (int(xyxy[0]), int(xyxy[1])),   # 左上坐标
                (int(xyxy[0]), int(xyxy[3])),   # 右上坐标
                (int(xyxy[2]), int(xyxy[3]))    # 右下坐标
            ]


def run():
    global window_points
    # 获取视频流
    cap = cv2.VideoCapture(r"./786ae0c4c19e880a7c931dd7cd1739d6.mp4")
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()
    # 读取视频帧
    ret, prev = cap.read()
    if not ret:
        print("Error: Cannot read video frame.")
        exit()

    # 初始化矩形框
    init_window_points(prev)

    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('warning') #创建窗口    
    cv2.setMouseCallback('warning', onMouse) #设置窗口鼠标回调函数 # type: ignore 

    flow = None
    idx = 0
    while True:
        ret, frame = cap.read()
        idx += 1
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.1, 0) # type: ignore
        prevgray = gray
        # 绘制光流线
        h, w = gray.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)#等差数列
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2).astype(np.int32)

        line = []
        movepoint=0        
        
        for l in lines:
            if l[0][0]-l[1][0]>3 or l[0][1]-l[1][1]>3:                
                line.append(l)
                if isinRec(l[0].tolist(), window_points):
                    movepoint+=1

        output = get_keypoints(frame)

        if movepoint > 20:  
            # print(movepoint + '注意！儿童进入危险区！')
            if 88 < idx < 144:
                print("危险！儿童正在爬窗！")
            else:
                print("注意！儿童进入危险区！")
            cv2.polylines(frame, line, 0, (255,255,0))
            pimg=drawRec(frame, window_points, True)  
        else:
            if 88 < idx < 144:
                print("危险！儿童正在爬窗！")
            else:
                print('安全')
            cv2.polylines(frame, line, 0, (0,255,0))
            pimg=drawRec(frame,window_points,False)              
              
        print(f"index = {idx}")
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(pimg, output[idx, 7:].T, 3)

        cv2.imshow('warning', pimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()       


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='dangerous_item.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./inference/images/786ae0c4c19e880a7c931dd7cd1739d6.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', default='./inference/result/', help='dir to save')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    run()
   