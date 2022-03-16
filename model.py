import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *

from numpy import * 

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.gen_batch_ops import batch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
#from ResNet import ResNet 
from keras.layers import Reshape


image_size = 512  # We'll resize input images to this size
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 6

projection = layers.Dense(units=projection_dim)
position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)



#SE操作：消除冗余信息聚集有效信息
def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

#基本卷积模块
def conv_block(inputs, filters):
    x = inputs

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = squeeze_excite_block(x)

    return x

class Patches(layers.Layer):
    def __init__(self, patch_size, name = "Patches", **kwargs):
        super(Patches, self).__init__(name = name, **kwargs)
        self.patch_size = patch_size
 
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):  #在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"patch_size": self.patch_size}
        base_config = super(Patches, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection, position_embedding, name = "PatchEncoder", **kwargs):
        super(PatchEncoder, self).__init__(name = name, **kwargs)
        self.num_patches = num_patches
        self.projection = projection
        self.position_embedding = position_embedding
        #self.projection = layers.Dense(units=projection_dim)
        #self.position_embedding = layers.Embedding(
        #    input_dim=num_patches, output_dim=projection_dim
        #)
 
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        #print("patch:",self.projection(patch))  #(None,None,64)
        #print("positions:",self.position_embedding(positions))  #(1024,64)
        #print("encoded:",encoded)   #(None,1024,64)
        return encoded

    def get_config(self):  #在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"num_patches":self.num_patches,"projection":self.projection,"position_embedding":self.position_embedding}
        base_config = super(PatchEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''
class Patches1(layers.Layer):
    def __init__(self, patch_size, name = "Patches1", **kwargs):
        super(Patches1, self).__init__(name = name, **kwargs)
        self.patch_size = patch_size
 
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):  #在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"patch_size": self.patch_size}
        base_config = super(Patches1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PatchEncoder1(layers.Layer):
    def __init__(self, num_patches, projection, position_embedding, name = "PatchEncoder1", **kwargs):
        super(PatchEncoder1, self).__init__(name = name, **kwargs)
        self.num_patches = num_patches
        self.projection = projection
        self.position_embedding = position_embedding
        #self.projection = layers.Dense(units=projection_dim)
        #self.position_embedding = layers.Embedding(
        #    input_dim=num_patches, output_dim=projection_dim
        #)
 
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        #print("patch:",self.projection(patch))  #(None,None,64)
        #print("positions:",self.position_embedding(positions))  #(1024,64)
        #print("encoded:",encoded)   #(None,1024,64)
        return encoded

    def get_config(self):  #在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"num_patches":self.num_patches,"projection":self.projection,"position_embedding":self.position_embedding}
        base_config = super(PatchEncoder1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def transformer1(x):
    # Augment data.     数据增强
    augmented = data_augmentation(x)
    # Create patches.      创造patch
    patches = Patches1(patch_size,name="Patches1")(augmented)
    encoded_patches = PatchEncoder1(num_patches, projection, position_embedding, name="PatchEncoder1")(patches)

    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    encoded_patches = Reshape((int(image_size/patch_size), int(image_size/patch_size), projection_dim))(encoded_patches)
    return encoded_patches
'''

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Normalization(),
        layers.experimental.preprocessing.Resizing(image_size, image_size),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(factor=0.02),
        layers.experimental.preprocessing.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)


def transformer(x):
    # Augment data.     数据增强
    augmented = data_augmentation(x)
    # Create patches.      创造patch
    patches = Patches(patch_size,name="Patches")(augmented)
    encoded_patches = PatchEncoder(num_patches, projection, position_embedding, name="PatchEncoder")(patches)

    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    encoded_patches = Reshape((int(image_size/patch_size), int(image_size/patch_size), projection_dim))(encoded_patches)
    return encoded_patches


#第一次编码采用VGG网络
def encoder1(inputs):
    skip_connections = []

    model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
    names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
    for name in names:
        skip_connections.append(model.get_layer(name).output)

    output = model.get_layer("block5_conv4").output
    return output, skip_connections

'''
#第一次编码采用VGG网络
def encoder3(inputs):
    model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    skip_connections = ResNet(inputs)
    output = model.get_layer("conv4_block6_out").output
    return output, skip_connections
'''



#正常解码
def decoder1(inputs, skip_connections):
    num_filters = [256, 128, 64, 32]
    skip_connections.reverse()
    x = inputs

    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_connections[i]])
        x = conv_block(x, f)

    return x


def encoder2(inputs):
    num_filters = [32, 64, 128, 256]
    skip_connections = []
    x = inputs

    for i, f in enumerate(num_filters):
        x = conv_block(x, f)
        skip_connections.append(x)
        x = MaxPool2D((2, 2))(x)

    return x, skip_connections


def decoder2(inputs, skip_1, skip_2):

    num_filters = [256, 128, 64, 32]
    skip_1.reverse() 

    skip_2.reverse()
    x = inputs

    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_1[i], skip_2[i]])
        x = conv_block(x, f)

    return x

def output_block(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    x = Activation('sigmoid')(x)
    return x

def Upsample(tensor, size):
    """Bilinear upsampling"""
    def _upsample(x, size):
        return tf.image.resize(images=x, size=size)
    return Lambda(lambda x: _upsample(x, size), output_shape=size)(tensor)

def ASPP(x, filter):
    shape = x.shape
    print(shape)
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)
    print("y1:",y1.shape)

    y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)
    print("y2:",y2.shape)

    y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)
    print("y3:",y3.shape)

    y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)
    print("y4:",y4.shape)

    y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)
    print("y5:",y5.shape)
    
    y = Concatenate()([y1, y2, y3, y4, y5])
    print("y",y.shape)

    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y

def Double_UNet(shape):
    #第一次编码和解码
    inputs = Input(shape)
    #print("input:",inputs.shape)
    x, skip_1 = encoder1(inputs)
    #print(x.shape)
    #x = transformer(x)
    x = ASPP(x, 64)
    x = decoder1(x, skip_1)
    outputs1 = output_block(x)

    #第二次编码的输入
    x = inputs * outputs1

    #第二次编码和解码
    x, skip_2 = encoder2(x)
    #print(x.shape)
    x = transformer(x)
    #x = ASPP(x, 64)
    x = decoder2(x, skip_1, skip_2)
    outputs2 = output_block(x)

    #两次输出进行连接
    outputs = Concatenate()([outputs1, outputs2])

    model = Model(inputs, outputs)
    return model

def MSSA_Net(shape):
    #第一次编码和解码
    inputs = Input(shape)
    x1, skip_1 = encoder1(inputs)
    x1 = ASPP(x1, 64)

    x2, skip_2 = encoder2(inputs)
    x2 = transformer(x2)
    #x3, skip_3 = encoder3(inputs)
    
    x = Concatenate()([x1,x2])
    #print("final:",x.shape)
    
    x = Conv2D(64, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    #print("fin:",x.shape)
    #print(x)
    #print(skip_1)
    #print(skip_2)
    x = decoder2(x,skip_1,skip_2)
    #x = decoder3(x,skip_1,skip_2,skip_3)
    outputs = output_block(x)

    outputs = Concatenate()([outputs, outputs])
    model = Model(inputs, outputs)
    return model

def TransUNet(shape):
    inputs = Input(shape)
    x, skip_1 = encoder1(inputs)
    x = transformer(x)
    x = decoder1(x, skip_1)
    outputs1 = output_block(x)

    outputs = Concatenate()([outputs1, outputs1])

    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":
    #model = encoder3(Input((512,512,3)))
    model = MSSA_Net((512,512,3))
    model.summary()
