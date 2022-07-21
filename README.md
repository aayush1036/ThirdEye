# ThirdEye 

### Introduction

This project aims to create an object detection model with common household objects and use text to speech to say the outputs aloud. This makes it very helpful for blind people to get information about their surroundings. 

Data is sourced from <a href="https://ieee-dataport.org/open-access/annotated-image-dataset-household-objects-robofeihome-team">Robofei@home team</a>

This project uses the <a href="https://github.com/ultralytics/yolov5">ultralytics yolov5 model</a>

More data can be added to the project by using the ```add_data.py``` script with the necessary command line arguments which captures images of a certain object using the webcam or downloads data from a given google drive link and saves it to a certain folder 

You can directly start training the model by using the ```train_model.py```  script with the necessary command line arguments if you have the ultralytics yolov5 model cloned on your device or else you can use the ```setup.py``` script to clone the model and install the required dependencies 

To perform live detections using a webcam, you can use the ```predict.py``` script with the necessary command line arguments

### Installation and setup 

Clone the repository by using 
``` git clone https://github.com/aayush1036/ThirdEye.git```

Create config.json file for roboflow 
```json
{
    "api_key":"YOUR_ROBOFLOW_API_KEY",
    "workspace":"YOUR_ROBOFLOW_WORKSPACE",
    "project":"YOUR_ROBOFLOW_PROJECT",
    "version":"YOUR_ROBOFLOW_DATA_VERSION"
}

```
Run the setup script to clone the model and install dependencies 

```python setup.py```

If you want to train the model then run the training script

```python train.py```

Then run the prediction script to perform real time detections

```python predict.py```