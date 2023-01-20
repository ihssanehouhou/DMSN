#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 2018

@author: longang
"""

import keras
from keras.models import Model
from keras.layers import Input, Dropout, Activation, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Cropping2D, UpSampling2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import concatenate, add, multiply, Subtract
from my_upsampling_2d import MyUpSampling2D
from instance_normalization import InstanceNormalization
import keras.backend as K
import tensorflow as tf

def loss(y_true, y_pred):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def loss2(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc2(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

class DMSN_module(object):
    
    def __init__(self, lr, img_shape, scene, vgg_weights_path):
        self.lr = lr
        self.img_shape = img_shape
        self.scene = scene
        self.vgg_weights_path = vgg_weights_path
        self.method_name = 'DMSN'
        
    def VGG16(self, x): 
        
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        F1 = x
    
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        F2 = x
    
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        F3 = x
    
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        F4 = x

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        F5 = x
        
        return F1, F2, F3, F4, F5

    def VGG16_Depth(self, x): 
        
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='d_block1_conv1', data_format='channels_last')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='d_block1_conv2')(x)
        
        x = MaxPooling2D((2, 2), strides=(2, 2), name='d_block1_pool')(x)
        d_F1 = x
    
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='d_block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='d_block2_conv2')(x)
        
        x = MaxPooling2D((2, 2), strides=(2, 2), name='d_block2_pool')(x)
        d_F2 = x
    
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='d_block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='d_block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='d_block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='d_block3_pool')(x)
        d_F3 = x
    
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='d_block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='d_block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='d_block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='d_block4_pool')(x)
        d_F4 = x

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='d_block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='d_block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='d_block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='d_block5_pool')(x)
        d_F5 = x
        
        return d_F1, d_F2, d_F3, d_F4, d_F5
    
    def decoder(self, X1, X2, X3, X4, X5, C1, C2, C3, C4, C5, d_F1, d_F2, d_F3, d_F4, d_F5):
        DC5 = concatenate([X5, C5, d_F5], name='DC_X5_C5')
        D5 = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same', name='D5')(DC5)

        DC4 = concatenate([X4, C4, d_F4, D5], name='DC_X4_C4')
        D4 = Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='D4')(DC4)

        DC3 = concatenate([X3, C3, d_F3, D4], name='DC_X3_C3')
        D3 = Conv2DTranspose(384, (3, 3), strides=(2, 2), activation='relu', padding='same', name='D3')(DC3)

        DC2 = concatenate([X2, C2, d_F2, D3], name='DC_X2_C2')
        D2 = Conv2DTranspose(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='D2')(DC2)

        DC1 = concatenate([X1, C1, d_F1, D2], name='DC_X1_C1')
        D1 = Conv2DTranspose(640, (3, 3), strides=(2, 2), activation='relu', padding='same', name='D1')(DC1)

        x = Conv2D(1, 1, strides=(1, 1), activation='sigmoid')(D1)
        return x
    
    
    #InterMediat Skeep Connection
    def IMSC(self, F1, F2, F3, F4, F5):

        X1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='X1_conv')(F1)
        AvX1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='APX1')(X1)
        C1 = Subtract()([X1, AvX1])
        X2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='X2_conv')(F2)
        AvX2 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='APX2')(X2)
        C2 = Subtract()([X2, AvX2])
        X3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='X3_conv')(F3)
        AvX3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='APX3')(X3)
        C3 = Subtract()([X3, AvX3])
        X4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='X4_conv')(F4)
        AvX4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='APX4')(X4)
        C4 = Subtract()([X4, AvX4])
        X5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='X5_conv')(F5)
        AvX5 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='APX5')(X5)
        C5 = Subtract()([X5, AvX5])
        
        
        return X1, X2, X3, X4, X5, C1, C2, C3, C4, C5    
    
    def initModel(self, dataset_name):
        assert dataset_name in ['CDnet', 'SBI', 'UCSD', 'SBM'], 'dataset_name must be either one in ["CDnet", "SBI", "UCSD", "SBM"]]'
        assert len(self.img_shape)==3
        h, w, d = self.img_shape
        
        net_input = Input(shape=(h, w, d), name='net_input')
        vgg_output = self.VGG16(net_input)
        model = Model(inputs=net_input, outputs=vgg_output, name='model')
        model.load_weights(self.vgg_weights_path, by_name=True)
        
        unfreeze_layers = ['block5_conv1','block5_conv2', 'block5_conv3']
        for layer in model.layers:
            if(layer.name not in unfreeze_layers):
                layer.trainable = False
                
        F1, F2, F3, F4, F5 = model.output

        net_input_depth = Input(shape=(h, w, 1), name='net_input_depth')
        vgg_output_depth = self.VGG16_Depth(net_input_depth)
        model_depth = Model(inputs=net_input_depth, outputs=vgg_output_depth, name='model_depth')
        model_depth.load_weights(self.vgg_weights_path, by_name=True)
        
        unfreeze_layers = ['d_block5_conv1','d_block5_conv2', 'd_block5_conv3']
        for layer in model_depth.layers:
            if(layer.name not in unfreeze_layers):
                layer.trainable = False
                
        d_F1, d_F2, d_F3, d_F4, d_F5 = model_depth.output
        
        # pad in case of CDnet2014
        if dataset_name=='CDnet':
            x1_ups = {'streetCornerAtNight':(0,1), 'tramStation':(1,0), 'turbulence2':(1,0)}
            for key, val in x1_ups.items():
                if self.scene==key:
                    # upscale by adding number of pixels to each dim.
                    x = MyUpSampling2D(size=(1,1), num_pixels=val, method_name = self.method_name)(x)
                    break
                
        X1, X2, X3, X4, X5, C1, C2, C3, C4, C5 = self.IMSC(F1, F2, F3, F4, F5)
        # d_X1, d_X2, d_X3, d_X4, d_X5, d_C1, d_C2, d_C3, d_C4, d_C5 = self.IMSC_D(d_F1, d_F2, d_F3, d_F4, d_F5)
        x = self.decoder(X1, X2, X3, X4, X5, C1, C2, C3, C4, C5, d_F1, d_F2, d_F3, d_F4, d_F5)
        
        # pad in case of CDnet2014
        if dataset_name=='CDnet':
            if(self.scene=='tramCrossroad_1fps'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0), method_name=self.method_name)(x)
            elif(self.scene=='bridgeEntry'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,2), method_name=self.method_name)(x)
            elif(self.scene=='fluidHighway'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0), method_name=self.method_name)(x)
            elif(self.scene=='streetCornerAtNight'): 
                x = MyUpSampling2D(size=(1,1), num_pixels=(1,0), method_name=self.method_name)(x)
                x = Cropping2D(cropping=((0, 0),(0, 1)))(x)
            elif(self.scene=='tramStation'):  
                x = Cropping2D(cropping=((1, 0),(0, 0)))(x)
            elif(self.scene=='twoPositionPTZCam'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(0,2), method_name=self.method_name)(x)
            elif(self.scene=='turbulence2'):
                x = Cropping2D(cropping=((1, 0),(0, 0)))(x)
                x = MyUpSampling2D(size=(1,1), num_pixels=(0,1), method_name=self.method_name)(x)
            elif(self.scene=='turbulence3'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0), method_name=self.method_name)(x)

        vision_model = Model(inputs=[net_input, net_input_depth], outputs=x, name='vision_model')
        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.)
        
        # Since UCSD has no void label, we do not need to filter out
        if dataset_name == 'UCSD':
            c_loss = loss2
            c_acc = acc2
        else:
            c_loss = loss
            c_acc = acc
        
        vision_model.compile(loss=c_loss, optimizer=opt, metrics=[c_acc])
        return vision_model