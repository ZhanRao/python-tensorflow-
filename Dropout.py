import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# 载入数据
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

# 每个批次的大小
batch_size=100;

# 计算一共有多少个批次
n_batch=mnist.train.num_examples//batch_size
# .train.num_examples表示数量，表示整除

# 定义两个placeholder
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)

# 创建一个简单的神经网路
W1=tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1=tf.Variable(tf.zeros([2000])+0.1)
L1=tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop=tf.nn.dropout(L1,keep_prob)

W2=tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2=tf.Variable(tf.zeros([2000])+0.1)
L2=tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop=tf.nn.dropout(L2,keep_prob)

W3=tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3=tf.Variable(tf.zeros([1000])+0.1)
L3=tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop=tf.nn.dropout(L3,keep_prob)

W4=tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4=tf.Variable(tf.zeros([10])+0.1)
prediction =tf.nn.softmax(tf.matmul(L3_drop,W4)+b4)

#二次代价函数
#loss=tf.reduce_mean(tf.square(y-prediction))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#使用梯度下降的法
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init =tf.global_variables_initializer()

#结果存放砸在一个布尔型列表中
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#argmax一个一维张量返回概率最大的位置

#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# tf.cast将布尔型数字转化为浮点型

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(20):
        for batch in range(n_batch):
            batch_xs,batchys=mnist.train.next_batch(batch_size)#每层只读取batch_size张
            # batch_xs保存图片的数据
            # batchys保存图片的标签
            sess.run(train_step,feed_dict={x:batch_xs,y:batchys,keep_prob:1.0})
        test_accy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels,keep_prob:1.0})
        train_accy=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        print("Iter"+str(epoch)+"   ,testig Accuracy"+str(test_accy)+"   ,train_Accuracy"+str(train_accy))

