# CASIA-Handwritten-Chinese-Character-Recognition

Get the idea from these two blogs:https://cloud.tencent.com/developer/article/1016464 AND
   https://coding.tools/blog/casia-handwritten-chinese-character-recognition-using-convolutional-neural-network-and-similarity-ranking


# Usage
Get dataset from CASIA, details in http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html

Train dataset download url: http://www.nlpr.ia.ac.cn/databases/Download/feature_data/1.0train-gb1.rar

Test dataset download url: http://www.nlpr.ia.ac.cn/databases/Download/feature_data/1.0test-gb1.rar

3 steps to implement classify a subset of CASIA database:

1. convert .gnt to .png which contains gray-scale value of pixes; then, store the values in arrays.

2.design CNN: 5 convolutional layers, 5 pooling layers, 1 fully connected layer.

3.design loss function and calculate accuracy.

# Result:
Loss:
![image](https://github.com/JinliangZ/CASIA-Handwritten-Chinese-Character-Recognition/blob/master/image/loss.jpg)


Model_softmax average accuracy: 0.937968


Flow graph:
![image](https://github.com/JinliangZ/CASIA-Handwritten-Chinese-Character-Recognition/blob/master/image/flow%20diagram.png)

