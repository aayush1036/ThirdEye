# Importing dependencies 
import argparse 
import cv2 
import sys 
import os 
import time 
import uuid 
from tqdm import tqdm 
import gdown

# Parsing command line arguments 
category = None 

parser = argparse.ArgumentParser()
parser.add_argument('--dst', help='Destination where the image needs to be saved', default='../Images')
parser.add_argument('--category', help='Category of the image')
parser.add_argument('--n_img', help='Number of images needed to be captured',default=50)
parser.add_argument('--delay', help='Delay time (in seconds) between capturing images', default=1)
parser.add_argument('--gdrive', help='Google drive link for folder',default=None)
parser.add_argument('--device', help="""The device you want to use for capturing the data, 0 for webcam 
                    otherwise the number of device plugged in to your PC""", default=0)

args = parser.parse_args()

dst = args.dst 
category = args.category 
delay = int(args.delay )
n_img = int(args.n_img)
gdrive_link = args.gdrive
device = int(args.device)

# Raising error if category is not specified  
if category is None:
    sys.exit('Usage python capture_images.py --category CATEGORY')

save_path = os.path.join(dst, category)
    
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print('Made folder for saving Images')
# Downloading the data from google drive if drive link is specified 
if gdrive_link is not None:
    gdown.download_folder(url=gdrive_link, output=save_path)

else:
    # Capturing the inputs and saving them 
    cap = cv2.VideoCapture(device)
    for i in tqdm(range(n_img)):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        img_name = uuid.uuid1()
        img_save_path = os.path.join(save_path, f'{img_name}.jpg')
        cv2.imwrite(img_save_path, frame)
        cv2.imshow('frame', frame)
        time.sleep(delay)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

print('Please annotate the data using RoboFlow and train the model on new data')