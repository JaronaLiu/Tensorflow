#-*- coding: utf-8 -*-
from skimage import io,transform
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

plt.rcParams['font.sans-serif']=['SimHei']


path1 = "F:/Work/python/data/test_12/马铃薯早疫病/04ee51b6-07e2-4182-84f8-46b22c8938a2___RS_Early.B 8091.jpg"
path2 = "F:/Work/python/data/test_12/玉米普通锈病/RS_Rust 1764.jpg"
path3 = "F:/Work/python/data/new_three/validation/00/wkb_0_8928.jpeg"
path4 = "F:/Work/python/data/new_three/validation/00/dwb_0_819.jpeg"
path5 = "F:/Work/python/data/test_12/番茄细菌性斑点病/2e604e19-fcb1-4d3c-9c04-4b154e1ca480___UF.GRC_BS_Lab Leaf 9241.jpg"
path6 = "F:/Work/python/data/test_12/玉米灰斑病/0f08ab4f-caf6-4e50-a399-000e4e9841a0___RS_GLSp 4623.jpg"
path7 = "F:/Work/python/data/test_12/玉米灰斑病/0dbbcb82-756e-48a6-94f4-3a8f1e1cedda___RS_GLSp 4587.jpg"
path8 = "F:/Work/python/data/test_12/马铃薯晚疫病/c7982d1f-7472-44ba-a39d-76085c4d90cd___RS_LB 3072.jpg"
path9 = "F:/Work/python/data/test_12/玉米大斑病/4b79b80a-6ad9-41b3-9d6a-7b493c5c75a0___RS_NLB 4258.jpg"

disease_dict = {0:'樱桃白粉病',1:'水稻稻曲病',2:'水稻稻瘟病',3:'水稻纹枯病',4:'玉米大斑病',5:'玉米普通锈病',6:'玉米灰斑病',7:'番茄细菌性斑点病',8:'苹果黑星病',9:'苹果黑腐病',10:'马铃薯早疫病',11:'马铃薯晚疫病'}

w=64
h=64
c=3

def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)

with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)
    data2 = read_one_image(path2)
    data3 = read_one_image(path3)
    data4 = read_one_image(path4)
    data5 = read_one_image(path5)
    data6 = read_one_image(path6)
    data7 = read_one_image(path7)
    data8 = read_one_image(path8)
    data9 = read_one_image(path9)
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)
    data.append(data6)
    data.append(data7)
    data.append(data8)
    data.append(data9)
#导入训练好的模型
    saver = tf.train.import_meta_graph('F:/Work/python/model/train_12_9.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('F:/Work/python/model/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)

    #打印出预测矩阵
   # print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result,1).eval())
    #根据索引通过字典对应病的分类
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("第",i+1,"张图预测结果: "+disease_dict[output[i]])


plt.figure("test images")

img1=Image.open(path1)
img2=Image.open(path2)
img3=Image.open(path3)
img4=Image.open(path4)
img5=Image.open(path5)
img6=Image.open(path6)
img7=Image.open(path7)
img8=Image.open(path8)
img9=Image.open(path9)

plt.subplot(3,3,1);plt.imshow(img1);plt.title(u'图一：马铃薯早疫病');plt.axis('off')

plt.subplot(3,3,2);plt.imshow(img2);plt.title(u'图二：玉米普通锈病');plt.axis('off')

plt.subplot(3,3,3);plt.imshow(img3);plt.title(u'图三：水稻稻瘟病');plt.axis('off')

plt.subplot(3,3,4);plt.imshow(img4);plt.title(u'图四：水稻稻瘟病');plt.axis('off')

plt.subplot(3,3,5);plt.imshow(img5);plt.title(u'图五：番茄细菌斑点病');plt.axis('off')

plt.subplot(3,3,6);plt.imshow(img6);plt.title(u'图六：玉米灰斑病');plt.axis('off')

plt.subplot(3,3,7);plt.imshow(img7);plt.title(u'图七：玉米灰斑病');plt.axis('off')

plt.subplot(3,3,8);plt.imshow(img8);plt.title(u'图八：马铃薯晚疫病');plt.axis('off')

plt.subplot(3,3,9);plt.imshow(img9);plt.title(u'图九：玉米大斑病');plt.axis('off')

plt.show()