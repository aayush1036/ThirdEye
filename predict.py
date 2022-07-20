# Importing dependencies 
import os
import torch 
import cv2 
import numpy as np 
import argparse

# Processing necessary command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, help="""The device with which you would like to capture the video
                    0 for webcam, otherwise the number of the video capturing device attached to your PC""")
parser.add_argument('--version', default=-1, help="""The version of model you would like to use to perform the detections, 
                    -1 for latest, otherwise enter the version number""")
args = parser.parse_args()
device = int(args.device)
version = int(args.version)

yolo_base_weights_path = os.path.join('yolov5','yolov5s.pt')
# Configuring the path where trained model is stored 
runs_path = os.path.join('yolov5', 'runs', 'train')

runs = os.listdir(runs_path)
# Finding out the latest version
if version == -1:
    latest_run = runs[version]
else:
    # if version is specified, using that model
    latest_run = runs[version-1]

# Loading the custom model using torch hub

# model = torch.hub.load('ultralytics/yolov5', 'custom', 
#                        path=os.path.join(runs_path, latest_run, 'weights', 'best.pt'), 
#                        force_reload=True)

model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_base_weights_path, force_reload=True)

# Performing real time detection using opencv 
cap = cv2.VideoCapture(device)
while cap.isOpened():
    ret, frame = cap.read()
    # Make detections 
    results = model(frame)
    labels = results.pandas().xyxy[0]['name'].values
    cv2.imshow('YOLO', np.squeeze(results.render()))
  
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()