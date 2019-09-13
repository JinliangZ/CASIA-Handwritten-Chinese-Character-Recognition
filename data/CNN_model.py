# -*- coding: utf-8 -*-
import tensorflow as tf
import DataManager as dm


#卷积层负责提取图像中的局部特征；
#池化层用来大幅降低参数量级(降维),同时改善结果（不易出现过拟合）；
#全连接层类似传统神经网络的部分，用来输出想要的结果。
X = tf.compat.v1.placeholder("float", [None, 64* 64])
Y = tf.compat.v1.placeholder("float", [None, 150])

def CNN(X,Y):
    input_layer = tf.reshape(X, shape=[-1, 64, 64, 1])
    # Input Tensor Shape: [batch_size, 64, 64, 1]
    # Output Tensor Shape: [batch_size, 64, 64, 32]
    with tf.name_scope("conv1"):
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5,5], strides=(1, 1), padding="same",
                                 activation=tf.nn.leaky_relu)  #避免梯度消失，收敛速度快
 
    # Input Tensor Shape: [batch_size, 64, 64, 32]
    # Output Tensor Shape: [batch_size, 32, 32, 32]
    with tf.name_scope("pool1"):
        pool1 = tf.layers.max_pooling2d (inputs=conv1, pool_size=[2,2], strides = 2) #值最大代表只保留这些特征中最强的，抛弃其他弱的此类特征。
 

    # Input Tensor Shape: [batch_size, 32, 32, 32]
    # Output Tensor Shape: [batch_size, 32, 32, 64]
    with tf.name_scope("conv2"):
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5,5], strides=(1, 1), padding="same", 
                                 activation=tf.nn.leaky_relu)  

    # Input Tensor Shape: [batch_size, 32, 32, 64]
    # Output Tensor Shape: [batch_size, 16, 16, 64]
    with tf.name_scope("pool2"):
        pool2 = tf.layers.max_pooling2d (inputs=conv2, pool_size=[2,2], strides = 2)
         

    # Input Tensor Shape: [batch_size, 16, 16, 64]
    # Output Tensor Shape: [batch_size, 16, 16, 128]
    with tf.name_scope("conv3"):
        conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[5,5], strides=(1, 1), padding="same", 
                                 activation=tf.nn.leaky_relu)  
        

    # Input Tensor Shape: [batch_size, 16, 16, 128]
    # Output Tensor Shape: [batch_size, 8, 8, 128]
    with tf.name_scope("pool3"):
        pool3 = tf.layers.max_pooling2d (inputs=conv3, pool_size=[2,2], strides = 2)
        
    
    # Input Tensor Shape: [batch_size, 8, 8, 128]
    # Output Tensor Shape: [batch_size, 8, 8, 256]
    with tf.name_scope("conv4"):
        conv4 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[5,5], strides=(1, 1), padding="same", 
                                 activation=tf.nn.leaky_relu)  
        
   
    # Input Tensor Shape: [batch_size, 8, 8, 256]
    # Output Tensor Shape: [batch_size, 4, 4, 256]
    with tf.name_scope("pool4"):
        pool4 = tf.layers.max_pooling2d (inputs=conv4, pool_size=[2,2], strides = 2)
        
    # Input Tensor Shape: [batch_size, 4, 4, 256]
    # Output Tensor Shape: [batch_size, 4, 4, 512]
    with tf.name_scope("conv5"):
        conv5 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[5,5], strides=(1, 1), padding="same", 
                                 activation=tf.nn.leaky_relu)  
        
   
    # Input Tensor Shape: [batch_size, 4, 4, 512]
    # Output Tensor Shape: [batch_size, 2, 2, 512]
    with tf.name_scope("pool5"):
        pool5 = tf.layers.max_pooling2d (inputs=conv5, pool_size=[2,2], strides = 2)
        
    # Input Tensor Shape: [batch_size, 2, 2, 512]
    # Output Tensor Shape: [batch_size, 2 * 2 * 512]
    feature_layer = tf.reshape(pool5, [-1, 2 * 2 * 512])
    
    return feature_layer        


def full_connected_layer(feature_layer):
    with tf.name_scope("dense1"):
        dense1 = tf.layers.dense(inputs=feature_layer, units=1024, activation=tf.nn.leaky_relu)
    with tf.name_scope("dropout1"):
        dropout1 = tf.layers.dropout(dense1, rate=0.25, training=dm.MODE == tf.estimator.ModeKeys.TRAIN)  #avoid overfitting
    with tf.name_scope("dense2"):
        dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.leaky_relu)
    with tf.name_scope("dropout2"):
        dropout2 = tf.layers.dropout(dense2, rate=0.25,  training=dm.MODE == tf.estimator.ModeKeys.TRAIN)
    with tf.name_scope("out_layer"):
        classification_layer = tf.layers.dense(inputs=dropout2, units=150, activation=tf.nn.leaky_relu)

    return classification_layer       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
             