# Importing dependencies 
import argparse 
import cv2 
import os 
import time 
import uuid 
from tqdm import tqdm 
import gdown

capture = True 
# Parsing command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--dst', help='Destination where the image needs to be saved', default='Captures/')
parser.add_argument('--n_img', help='Number of images needed to be captured',default=50)
parser.add_argument('--delay', help='Delay time (in seconds) between capturing images', default=1)
parser.add_argument('--gdrive', help='Google drive link for folder',default=None)
parser.add_argument('--device', help="""The device you want to use for capturing the data, 0 for webcam 
                    otherwise the number of device plugged in to your PC""", default=0)
parser.add_argument('--src', help='The video file from which you want to capture images', default=None)
parser.add_argument('--format', help='The file format in which you want to save the images', default='jpg')

args = parser.parse_args()

dst = args.dst 
delay = int(args.delay )
n_img = int(args.n_img)
gdrive_link = args.gdrive
device = int(args.device)
src = args.src 
file_format = args.format
    
# Making the destination folder if it does not exist 
if not os.path.exists(dst):
    os.makedirs(dst)
    print('Made folder for saving Images')

# Downloading the data from google drive if drive link is specified 
if gdrive_link is not None:
    capture = False 
    gdown.download_folder(url=gdrive_link, output=dst)    

if capture and src is None:
    # Capturing the inputs and saving them 
    cap = cv2.VideoCapture(device)
    for i in tqdm(range(n_img)):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        img_name = f'{uuid.uuid1()}.{file_format}'
        img_save_path = os.path.join(dst, img_name)
        cv2.imwrite(img_save_path, frame)
        cv2.imshow('frame', frame)
        time.sleep(delay)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

elif capture and src is not None:
    cap = cv2.VideoCapture(src)
    while cap.isOpened():
        ret, frame = cap.read()
        img_name = f'{uuid.uuid1()}.{file_format}'
        img_save_path = os.path.join(dst, img_name)
        cv2.imwrite(img_save_path, frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()    
    
print('Please annotate the data using RoboFlow and train the model on new data')