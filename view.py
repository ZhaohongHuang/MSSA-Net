import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from glob import glob
from sklearn.model_selection import train_test_split
from utils import *
from train import tf_dataset
from tqdm import tqdm
from PIL import Image

from metrics import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import Adam, Nadam
#os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

def read_image(x):
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = np.array(image)

    image = np.clip(image - np.median(image)+127, 0, 255)
    image = image/255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

def read_mask(y):
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = mask/255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask

def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def parse(y_pred):
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = y_pred[..., -1]
    y_pred = y_pred.astype(np.float32)
    y_pred = np.expand_dims(y_pred, axis=-1)
    return y_pred

def evaluate_normal(model1, model2, model3, model4, model5, path):
    THRESHOLD = 0.5
    total = []
    x = read_image(path)
    _, h, w, _ = x.shape

    y_pred1 = parse(model1.predict(x)[0][..., -1])
    y_pred2 = parse(model2.predict(x)[0][..., -1])
    y_pred3 = parse(model3.predict(x)[0][..., -1])
    y_pred4 = parse(model4.predict(x)[0][..., -1])
    y_pred5 = parse(model5.predict(x)[0][..., -1])

    y_pred1 = mask_to_3d(y_pred1) * 255.0
    y_pred2 = mask_to_3d(y_pred2) * 255.0
    y_pred3 = mask_to_3d(y_pred3) * 255.0
    y_pred4 = mask_to_3d(y_pred4) * 255.0
    y_pred5 = mask_to_3d(y_pred5) * 255.0

    cv2.imwrite("view/result/1.png", y_pred1)
    cv2.imwrite("view/result/2.png", y_pred2)
    cv2.imwrite("view/result/3.png", y_pred3)
    cv2.imwrite("view/result/4.png", y_pred4)
    cv2.imwrite("view/result/5.png", y_pred5)

    img1 = Image.open(path)
    img2 = Image.open("view/result/1.png")
    img3 = Image.open("view/result/2.png")
    img4 = Image.open("view/result/3.png")
    img5 = Image.open("view/result/4.png")
    img6 = Image.open("view/result/5.png")

    img1 = img1.convert("RGBA") 
    img2 = img2.convert("RGBA")   
    img3 = img3.convert("RGBA")   
    img4 = img4.convert("RGBA")   
    img5 = img5.convert("RGBA")   
    img6 = img6.convert("RGBA")   


    width, height = img2.size

    for i in range(0,width):
        for j in range(0,height):
            data = img2.getpixel((i,j))
            if (data.count(255) == 4):
                img1.putpixel((i,j),(255,0,0,255))

    img1.save("view/result/111.png")


    img1 = Image.open(path)
    img1 = img1.convert("RGBA") 
    for i in range(0,width):
        for j in range(0,height):
            data = img3.getpixel((i,j))
            if (data.count(255) == 4):
                img1.putpixel((i,j),(255,0,0,255))

    img1.save("view/result/222.png")

    img1 = Image.open(path)
    img1 = img1.convert("RGBA") 
    for i in range(0,width):
        for j in range(0,height):
            data = img4.getpixel((i,j))
            if (data.count(255) == 4):
                img1.putpixel((i,j),(255,0,0,255))

    img1.save("view/result/333.png")

    img1 = Image.open(path)
    img1 = img1.convert("RGBA") 
    for i in range(0,width):
        for j in range(0,height):
            data = img5.getpixel((i,j))
            if (data.count(255) == 4):
                img1.putpixel((i,j),(255,0,0,255))

    img1.save("view/result/444.png")
    
    img1 = Image.open(path)
    img1 = img1.convert("RGBA") 
    for i in range(0,width):
        for j in range(0,height):
            data = img6.getpixel((i,j))
            if (data.count(255) == 4):
                img1.putpixel((i,j),(255,0,0,255))

    img1.save("view/result/555.png")

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

if __name__ == "__main__":
    model1 = load_model_weight("files/MeNet_lite.h5")
    model2 = load_model_weight("files/cz_DoubleUNet.h5")
    model3 = load_model_weight("files/cz_TransUNet.h5")
    model4 = load_model_weight("files/cz_UNet.h5")
    model5 = load_model_weight("files/MeNet_MobileNet.h5")

    print("model already Finish !!!")

    path = "cz_data/classes/m/image/031958_t1w_deface_stx.nii.gz107.png.png"
    mask = "cz_data/classes/m/mask/031958_LesionSmooth_stx.nii.gz107.png.png"
    evaluate_normal(model1, model2, model3, model4, model5, path)

    image = Image.open(path)
    image = image.convert("RGBA")

    img6 = Image.open(mask)
    img6 = img6.convert("RGBA")

    width, height = image.size

    for i in range(0,width):
        for j in range(0,height):
            data = img6.getpixel((i,j))
            if (data.count(255) == 4):
                image.putpixel((i,j),(255,0,0,255))
    image.save("view/result/gt.png")
    
    gt = cv2.imread("view/result/gt.png")
    img1 = cv2.imread("view/result/111.png")
    img2 = cv2.imread("view/result/222.png")
    img3 = cv2.imread("view/result/333.png")
    img4 = cv2.imread("view/result/444.png")
    img5 = cv2.imread("view/result/555.png")

    h, w, _ = img1.shape

    line = np.ones((h, 10, 3)) * 255.0

    all_images = [
        gt, line,
        img1, line,
        img2, line,
        img3, line,
        img4, line,
        img5
        ]

    mask = np.concatenate(all_images, axis=1)

    cv2.imwrite("1.png",mask)