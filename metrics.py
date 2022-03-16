import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

smooth = 1e-15

def TP(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    return true_positives


def FP(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f01 = K.round(K.clip(y_pred_f, 0, 1))
    tp_f01 = K.round(K.clip(y_true_f * y_pred_f, 0, 1))
    false_positives = K.sum(K.round(K.clip(y_pred_f01 - tp_f01, 0, 1)))
    return false_positives


def TN(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f01 = K.round(K.clip(y_pred_f, 0, 1))
    all_one = K.ones_like(y_pred_f01)
    y_pred_f_1 = -1 * (y_pred_f01 - all_one)
    y_true_f_1 = -1 * (y_true_f - all_one)
    true_negatives = K.sum(K.round(K.clip(y_true_f_1 + y_pred_f_1, 0, 1)))
    return true_negatives


def FN(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # y_pred_f01 = keras.round(keras.clip(y_pred_f, 0, 1))
    tp_f01 = K.round(K.clip(y_true_f * y_pred_f, 0, 1))
    false_negatives = K.sum(K.round(K.clip(y_true_f - tp_f01, 0, 1)))
    return false_negatives

def DSC(y_ture, y_pred):
    DSC = (2 * TP(y_ture, y_pred) + smooth) / ((2 * TP(y_ture, y_pred) + FP(y_ture, y_pred) + FN(y_ture, y_pred))  + smooth)
    return DSC

def mIoU(y_ture, y_pred):
    mIoU = (TP(y_ture, y_pred) + smooth) / ((FN(y_ture, y_pred) + FP(y_ture, y_pred) + TP(y_ture, y_pred))  + smooth)
    return mIoU

def precision(y_ture, y_pred):
    Precision = (TP(y_ture, y_pred) + smooth) / ((FP(y_ture, y_pred) + TP(y_ture, y_pred))  + smooth)
    return Precision

def recall(y_ture, y_pred):
    Recall = (TP(y_ture, y_pred) + smooth) / ((FN(y_ture, y_pred) + TP(y_ture, y_pred))  + smooth)
    return Recall


def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + smooth) / (union + smooth)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def bce_dice_loss(y_true, y_pred):
    return (binary_crossentropy(y_true, y_pred) ** 2 + dice_loss(y_true, y_pred) ** 2) ** 0.5

def focal_loss(y_true, y_pred):
    alpha=0.25
    gamma=2
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))
    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)
