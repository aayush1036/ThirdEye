# Importing dependencies 
import os 
import argparse
# Fetching the path for yaml file which contains information about custom data labels
yaml_path = '../data.yaml'

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