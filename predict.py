# Importing dependencies 
import os
import torch 
import cv2
import numpy as np 
import argparse
import pyttsx3

# Processing necessary command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, help="""The device with which you would like to capture the video
                    0 for webcam, otherwise the number of the video capturing device attached to your PC""")
parser.add_argument('--version', default=0, help="""The version of model you would like to use to perform the detections, 
                    -1 for latest, 0 for pre trained, otherwise enter the version number""")
parser.add_argument('--frame_rate', default=10, help='The frame rate after which you would  like to hear outputs')

args = parser.parse_args()
device = int(args.device)
version = int(args.version)
frame_rate = int(args.frame_rate)

if version == 0:
    latest_run = 'best.pt'

else:
    yolo_base_weights_path = os.path.join('yolov5','yolov5s.pt')
    # Configuring the path where trained model is stored 
    runs_path = os.path.join('yolov5', 'runs', 'train')

    runs = os.listdir(runs_path)
    # Finding out the latest version
    if version == -1:
        latest_run = os.path.join(runs_path, runs[version], 'weights','best.pt')

    else:
        # if version is specified, using that model
        latest_run = os.path.join(runs_path, runs[version-1], 'weights','best.pt')

# Loading the custom model using torch hub

# model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join('yolov5', 'yolov5s.pt'), force_reload=True)

model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       path=latest_run, 
                       force_reload=True)
engine = pyttsx3.init()
# Performing real time detection using opencv 
cap = cv2.VideoCapture(device)
curr_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    curr_frame += 1
    # Make detections 
    results = model(frame)
    labels = results.pandas().xyxy[0]['name'].values
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if curr_frame % frame_rate == 0:
        for label in labels:
            engine.say(label)
            engine.runAndWait()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()