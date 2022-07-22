# Importing dependencies 
import os 
import argparse
from roboflow import Roboflow
import json 
import yaml 
import shutil

# Reading credential files 
with open('config.json', 'r')  as f:
    config = json.load(f)
    
api_key = config.get('api_key')
workspace = config.get('workspace')
project = config.get('project')
version = config.get('version')

# Downloading the data from RoboFlow 
rf = Roboflow(api_key=api_key)
project = rf.workspace(workspace).project(project)
dataset = project.version(version).download("yolov5")

yaml_path = os.path.join(dataset.location, 'data.yaml')

with open(yaml_path, 'r') as f:
    yaml_file = yaml.safe_load(f)
    
yaml_file['train'] = os.path.abspath(os.path.join(dataset.location, 'train'))
yaml_file['val'] = os.path.abspath(os.path.join(dataset.location, 'valid'))

with open(yaml_path, 'w') as f:
    yaml.dump(dict(yaml_file), f)

is_setup = os.path.exists('yolov5')
if not is_setup:
    os.system('python setup.py')
# Fetching the path where yolo model has been stored 
# Parsing some command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--img', default=416, help='Image size for which you want to train the model')
parser.add_argument('--batch', default=16, help='Batch size to train your model')
parser.add_argument('--epochs', default=500, help='Number of epochs to train your model')
parser.add_argument('--data', default=yaml_path, help='Path of the yaml file for training model on custom dataset')
parser.add_argument('--weights', default='yolov5s.pt', help='The model architecture you want to use, refer to https://github.com/ultralytics/yolov5#pretrained-checkpoints for more info')

args = parser.parse_args()

img = int(args.img)
batch = int(args.batch )
epochs = int(args.epochs)
data = args.data
weights = args.weights 
# Changing the working directory to the path where yolo repository is cloned 
# Creating the command for training the model 
train_command = f'cd yolov5 && python train.py --img {img} --batch {batch} --epochs {epochs} --data {data} --weights {weights}'
# Training the model 
os.system(train_command)

runs_path = os.path.join('yolov5', 'runs', 'train')
latest_run = os.listdir(runs_path)[-1]
latest_run_model = os.path.join(runs_path, latest_run, 'weights', 'best.pt')
# Copy latest model to home directory
shutil.copy2(src=latest_run_model, dst='best.pt')