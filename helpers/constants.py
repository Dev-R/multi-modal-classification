import os
import numpy as np
import webcolors


class ActionRecognition:
    IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
    MAX_IMAGES_PER_CLASS = 8000

    DATASET_DIRECTORY = "modules/video_classification_CNN/UCF_CRIME"
    CLASSES_LIST = os.listdir(DATASET_DIRECTORY)  # Return all classes name in UCF50
    MODEL = "modules/video_classification_CNN/Model___Date_Time_2023_01_23__22_37_31___Loss_0.5416110157966614___Accuracy_0.9529687762260437.h5"

    # Video classes that will trigger a red flag
    ANOMALY_CLASSES_NAME = ["Explosion"]

    TOP_CONFIDENCE_CLASS_TEXT_COLOR = webcolors.name_to_rgb("green")

    HIGH_CONFIDENCE_CLASS_TEXT_COLOR = webcolors.name_to_rgb("orange")

    MID_CONFIDENCE_CLASS_TEXT_COLOR = webcolors.name_to_rgb("yellow")

    LOW_CONFIDENCE_CLASS_TEXT_COLOR = webcolors.name_to_rgb("red")


class AudioRecognition:

    # Audi classes that will trigger a red flag
    ANOMALY_CLASSES_NAME = ["Explosion", "'Gunshot, gunfire'"]


class ObjectDetection:

    # Object detection_classes
    CLASSES = [
        "aeroplane",
        "background",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    PROTOTXT = "data/MobileNetSSD_deploy.prototxt.txt"
    MODEL = "data/MobileNetSSD_deploy.caffemodel"
    CONFIDENCE = 0.2
    # Assigning random colors to each of the classes
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


class Config:

    # Directory to save image path

    MEDIA_PATH = "outputs"
    NOTIFICATION_CONFIG = {"notification_type": "sms", "notification_interval": 60 * 2}
