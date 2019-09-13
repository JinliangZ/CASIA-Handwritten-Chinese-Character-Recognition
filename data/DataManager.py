# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.misc
from sklearn.utils import shuffle
import tensorflow as tf
#import gnt2png as gn
import struct


data_dir = '../data'
train_data_dir = os.path.join(data_dir, 'HWDB1.1trn_gnt')
test_data_dir = os.path.join(data_dir, 'HWDB1.1tst_gnt')

# 读取图像和对应的汉字
def read_from_gnt_dir(gnt_dir=train_data_dir):
	def one_file(f):
		header_size = 10
		while True:
			header = np.fromfile(f, dtype='uint8', count=header_size)
			if not header.size: break
			sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
			tagcode = header[5] + (header[4]<<8)
			width = header[6] + (header[7]<<8)
			height = header[8] + (header[9]<<8)
			if header_size + width*height != sample_size:
				break
			image = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
			yield image, tagcode
 
	for file_name in os.listdir(gnt_dir):
		if file_name.endswith('.gnt'):
			file_path = os.path.join(gnt_dir, file_name)
			with open(file_path, 'rb') as f:
				for image, tagcode in one_file(f):
					yield image, tagcode
 
    
    
# 我取常用的前150个汉字进行测试
char_set = "的一丁七是了我不人在他有这个上们来东到时大地为子中你丢说串生国年着就那和亿要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但低现前些所同日手张又行意动方侧期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高低已亲其进此话常与活正感"

def resize_and_normalize_image(img):
	# 补方
	pad_size = abs(img.shape[0]-img.shape[1]) // 2  #除2向下取整
	if img.shape[0] < img.shape[1]:
		pad_dims = ((pad_size, pad_size), (0, 0)) #上下补pad_size大小，左右不补
	else:
		pad_dims = ((0, 0), (pad_size, pad_size))
	img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255) #填充为白色
	# 缩放 将图片稍稍居中
	img = scipy.misc.imresize(img, (64 - 4*2, 64 - 4*2))
	img = np.lib.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=255)
	assert img.shape == (64, 64)
 
	img = img.flatten() #变成1维数组
	# 像素值范围-1到1
	img = (img - 128) / 128
	return img
 
# one hot
def convert_to_one_hot(char):
	vector = np.zeros(len(char_set))
	vector[char_set.index(char)] = 1 #.index查找到索引位置，将标签处设置为1，其余出为0
	return vector
 
# 由于数据量不大, 可一次全部加载到RAM
train_data_x = []
train_data_y = []

for image, tagcode in read_from_gnt_dir(gnt_dir= train_data_dir):
	tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312') 
	if tagcode_unicode in char_set:
		train_data_x.append(resize_and_normalize_image(image))
		train_data_y.append(convert_to_one_hot(tagcode_unicode))
 
# shuffle样本
train_data_x, train_data_y = shuffle(train_data_x, train_data_y)
 
batch_size = 128
num_batch = len(train_data_x) // batch_size
 

test_data_x = []
test_data_y = []
for image, tagcode in read_from_gnt_dir(gnt_dir= test_data_dir):
	tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
	if tagcode_unicode in char_set:
		test_data_x.append(resize_and_normalize_image(image))
		test_data_y.append(convert_to_one_hot(tagcode_unicode))
# shuffle样本
test_data_x, test_data_y = shuffle(test_data_x, test_data_y)
num = len(test_data_x) 
 
#X = tf.placeholder(tf.float32, [None, 64*64])
#Y = tf.placeholder(tf.float32, [None, 150])
#keep_prob = tf.placeholder(tf.float32)

#MODE = tf.estimator.ModeKeys.TRAIN
MODE = tf.estimator.ModeKeys.EVAL
 

