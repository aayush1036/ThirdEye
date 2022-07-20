# Data Source : https://ieee-dataport.org/open-access/annotated-image-dataset-household-objects-robofeihome-team 
import os 
import shutil
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import yaml 

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',default=os.path.join('dataset','Objects in the lab', 'images'))
parser.add_argument('--labels_path', default='labels/')
parser.add_argument('--test_size', default=0.1)
parser.add_argument('--val_size', default=0.1)
parser.add_argument('--yaml_path', default='data.yaml')
parser.add_argument('--destination', default='data')
parser.add_argument('--annotations_path', default=os.path.join('dataset','Objects in the lab', 'annotations'))

# parsing command line arguments 
args = parser.parse_args()
data_path = args.data_path 
labels_path = args.labels_path
test_size = float(args.test_size)
val_size = float(args.val_size)
yaml_path = args.yaml_path 
destination = args.destination
annotations_path = args.annotations_path

if not os.path.exists(labels_path):
    os.makedirs(labels_path)

files = os.listdir(data_path)
labels = os.listdir(labels_path)

# Checking if yolo labels exist or the data is properly labelled 
if (len(files)+1 != len(labels) and len(labels) !=0):
    for file in os.listdir(labels_path):
        os.remove(file)
if len(labels) == 0:
    command = f'python xml2yolo.py --input_dir "{annotations_path}" --image_dir "{data_path}" --output_dir "{labels_path}"'
    os.system(command)
    labels = os.listdir(labels_path)
# Splitting data
labels.remove('classes.txt')
assert len(files) == len(labels)
n_images = len(files)
test_size = int(test_size * n_images)
val_size = int(val_size * n_images)

files_train, files_test, labels_train, labels_test = train_test_split(files, labels, test_size=test_size)
files_train, files_val, labels_train, labels_val = train_test_split(files_train, labels_train, test_size=val_size)

# Creating train, test, val folders 
if os.path.exists(destination):
    # Removing any previous splits 
    shutil.rmtree(destination)
# Making folders for train, val and test sets 
os.makedirs(destination)
for subset in ['train', 'val', 'test']:
    subset_path = os.path.join(destination, subset)
    subset_images_path = os.path.join(subset_path, 'images')
    subset_labels_path = os.path.join(subset_path, 'labels')
    os.makedirs(subset_images_path)
    os.makedirs(subset_labels_path)  
    
path_dict = {}
train_set = zip(files_train, labels_train)
val_set = zip(files_val, labels_val)
test_set = zip(files_test, labels_test)
sets = {
    'train':train_set,
    'val':val_set,
    'test':test_set
}

for name, subset in sets.items():
    print(f'Moving files for {name}')
    for file, label in tqdm(subset):
        image_src = os.path.join(data_path, file)
        label_src = os.path.join(labels_path, label)
        image_dst = os.path.join(destination, name, 'images', file)
        label_dst = os.path.join(destination, name, 'labels', label)
        shutil.copy2(src=image_src, dst=image_dst)
        shutil.copy2(src=label_src, dst=label_dst)    
        
with open(os.path.join(labels_path, 'classes.txt'), 'r') as f:
    classes = f.read()
    
classes = classes.replace('[','')
classes = classes.replace(']', '')
classes = classes.replace('"', '')
classes = classes.split(',')

classes = [str(category.strip()) for category in classes]

data = {
    'train': os.path.abspath(os.path.join('data','train','images')),
    'val':os.path.abspath(os.path.join('data','val','images')),
    'names':classes,
    'nc':len(classes)
}

with open('data.yaml', 'w') as f:
    yaml.dump(data, f)