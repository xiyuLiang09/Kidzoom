# Kidzoom

Intelligent Child Home Monitoring System based on YOLO and Pose Recognition Models.

`KidZoom` KidZoom is an Android app developed based on `YOLOv7` (for dangerous object and posture detection), `Raspberry Pi` (camera and temperature/humidity sensors), and `front-end/back-end development`. We developed this software to prevent children👶 who are home alone from being harmed by dangerous objects, such as flames, scissors, and glass cups, due to a lack of safety awareness. Additionally, the system effectively detects children’s window-climbing behavior and promptly sends alert notifications to the parents' mobile application, helping parents who are away from home to better protect their children. 

## 🎬 Demo

![demo](https://github.com/user-attachments/assets/4bd9bcfb-0951-472c-b2bd-2ba6e10b48e1)

## 🎯 Key Features
- 🔥**Hazardous Object Detection**: Identifies dangerous items such as flames, scissors, and glass cups in the scene and sends alerts.
- 🚷**Electronic Fence**: Monitors restricted areas and provides alerts when boundaries are crossed.
- ⚠️**Climbing Behaviors Detection & Warning**: Recognizes climbing postures, calculates the overlap between the child and the window, and sends alert messages via the mobile app.
- 🌡️**Temperature and Humidity Monitoring**: Sensors detect and record environmental temperature and humidity continuously, issuing alerts when abnormal values are detected.  
- 🔅**Lighting Adaptation**: Automatically adjusts detection under varying lighting conditions for consistent performance.    
- 🕜**Real-Time Monitoring**: Ensures continuous surveillance and accurate hazard recognition.  

## Algorithms
- **YOLOv7**: Trained on a large, self-collected dataset of hazardous objects to improve the model's accuracy in recognizing these items. 
- **PP-TinyPose**: A lightweight and efficient pose estimation model, trained by us to detect window-climbing postures in real-time video streams.
- **SE-Net**: Considering the small spatial proportion of hazardous objects in real-world surveillance videos, this attention module is integrated into the YOLOv7 architecture to enhance the model's accuracy in detecting small objects.