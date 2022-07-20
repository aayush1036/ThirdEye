import os

clone_command = 'git clone https://github.com/ultralytics/yolov5.git'
os.system(clone_command)
requirements_command = 'cd yolov5 && pip install -r requirements.txt'
os.system(requirements_command)