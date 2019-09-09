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

# 创建一个简单的神经网路
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
prediction =tf.nn.softmax(tf.matmul(x,w)+b)

#二次代价函数
#loss=tf.reduce_mean(tf.square(y-prediction))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#使用梯度下降的法
#train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step=tf.train.AdadeltaOptimizer(0.02).minimize(loss)

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
    for epoch in range(40):
        for batch in range(n_batch):
            batch_xs,batchys=mnist.train.next_batch(batch_size)#每层只读取batch_size张
            # batch_xs保存图片的数据
            # batchys保存图片的标签
            sess.run(train_step,feed_dict={x:batch_xs,y:batchys})
        accy=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter"+str(epoch)+"   ,testig Accuracy"+str(accy))
    # plt.figure()
    # plt.plot(accy,'r-',lw=5)
    # plt.show()
