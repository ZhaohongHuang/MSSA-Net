import os
import numpy as np
import cv2
import json
from glob import glob
from metrics import *
from sklearn.utils import shuffle
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.models import load_model
from LGTA import *
#from model import build_model, Upsample, ASPP,Patches,Patches1,PatchEncoder,PatchEncoder1

def relu6(x):
    return K.relu(x, max_value=6)

_custom_objects = {
    "relu6" :  relu6,
    "Patches": Patches,
    "PatchEncoder": PatchEncoder
}

def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")

def read_data(x, y):
    """ Read the image and mask from the given path. """
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = cv2.resize(image,[512,512])
    mask = cv2.imread(y, cv2.IMREAD_COLOR)
    mask = cv2.resize(mask,[512,512])
    return image, mask

def read_params():
    """ Reading the parameters from the JSON file."""
    with open("params.json", "r") as f:
        data = f.read()
        params = json.loads(data)
        return params

def load_data(path):
    """ Loading the data from the given path. """
    images_path = os.path.join(path, "image/*")
    masks_path  = os.path.join(path, "mask/*")

    images = glob(images_path)
    masks  = glob(masks_path)

    return images, masks

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_model_weight(path):
    with CustomObjectScope({
        'dice_loss': dice_loss,
        'dice_coef': dice_coef,
        'bce_dice_loss': bce_dice_loss,
        'focal_loss': focal_loss,
        'iou': iou,
        'TP': TP, 'TN': TN, 'FN': FN, 'FP': FP,
        'DSC': DSC,
        'mIoU': mIoU,
        "precision": precision,
        "recall": recall
        }):
        model = load_model(path,custom_objects=_custom_objects)
    return model
    # model = build_model(256)
    # model.load_weights(path)
    # return model
