# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:30:35 2019

@author: Administrator
"""
import tensorflow as tf
import random
import numpy as np
import DataManager as dm, Logger, CNN_model

log = Logger.get_logger("Model_1","./log/Model_1.log")
weight_path = './weight/Model_1/Model_1.ckpt'

# tf Graph input
X = tf.placeholder("float", [None, 64*64])
Y = tf.placeholder("float", [None, 150])

#use convolutional neural network to extract features
feature_layer = CNN_model.CNN(X, Y)

#use fully connected neural network to classify
classification_layer = CNN_model.full_connected_layer(feature_layer)

# loss_softmax is the only loss function used in model 1
loss_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = classification_layer, labels = Y ))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1)
train_op = optimizer.minimize(loss_softmax)

# Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    saver = tf.train.Saver()  
    try:
        saver.restore(sess, weight_path) 
    except:
        log.info("no saved weight found.")
        
        
        
    if (dm.MODE == tf.estimator.ModeKeys.TRAIN):
            log.info("start training model...")   
            for epoch in range(50):
                for i in range(dm.num_batch):
                    batch_x = dm.train_data_x[i*dm.batch_size : (i+1)*dm.batch_size ]
                    batch_y = dm.train_data_y[i*dm.batch_size : (i+1)*dm.batch_size ]
                    _, cost = sess.run([train_op, loss_softmax], feed_dict={X:batch_x, Y:batch_y})
                    if np.mod(i, 100) == 0:
                        save_path = saver.save(sess, weight_path)
                        log.info('save weight')
                    
                    
                    
                    
    if (dm.MODE == tf.estimator.ModeKeys.EVAL):
            log.info("start evaluate model...")
            test_sample_total_accuracy = 0
            for epoch in range(100):
                i = random.randint(dm.num,dm.num-500)
                batch_x = dm.test_data_x[i : i+500]
                batch_y = dm.test_data_y[i : i+500]
    
                pred = tf.nn.softmax(classification_layer)  #转化成概率值
                
                correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(Y, axis=1)) #返回每一行最大值的索引与标签做判断返回对应的布尔值
                
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) #讲布尔值转化为0，1
                
                accuracy_eval = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
                
                log.info("model accuracy in this test batch: %f" % accuracy_eval)                
                
                test_sample_total_accuracy = test_sample_total_accuracy + accuracy_eval
                
            test_sample_average_accuracy = test_sample_total_accuracy / epoch
            
            log.info("finished testing model.")
            
            log.info("average model accuracy of this test dataset: %f" % test_sample_average_accuracy)    
            
    
    # TensorBoard
    tf.summary.scalar("loss", loss_softmax)
    tf.summary.scalar("accuracy", test_sample_average_accuracy) 
    merged_summary_op = tf.summaries.merge_all()
    writer = tf.summary.FileWriter("./graph",sess.graph)
        
        
        
        









                