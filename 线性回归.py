import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#函数的是逐渐按倒推的

# 使用numpy生成200个随机点
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
# 定义-0.5到0.5的200个数
# 如果没有[:,np.newaxis]将会变成一行
# [:,np.newaxis]可以将数据变成一列
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

# 定义两个placeholder
x=tf.placeholder(tf.float32,[None,1])
# 行数不能保证，但是保证列数只有一个
y=tf.placeholder(tf.float32,[None,1])
# 行数不能保证，但是保证列数只有一个

# 定义神经网络的中间层
weight_l1=tf.Variable(tf.random.normal([1,10]))
# 因为输入为1，中间层为10
# tf.random.normal第一个参数表示体积，而且一般用于w、b的定义
# np.random.normal()第一个平均值，第二个方差，第三个体积，一般用来定义初始值

biase_L1=tf.Variable(tf.zeros([1,10]))
# 因为输入为1，中间层为10
# 所有函数调用都要用（[]）
Wx_plus_L1=tf.matmul(x,weight_l1)+biase_L1
#相乘过后的输出为一行
L1=tf.nn.tanh(Wx_plus_L1)
# 激活函数，L1相当于中间层的输入

# 定义神经网络的输出
weight_L2=tf.Variable(tf.random_normal([10,1]))
biase_L2=tf.Variable(tf.zeros([1,1]))
wx_plus_L2=tf.matmul(L1,weight_L2)+biase_L2
prediction=tf.nn.tanh(wx_plus_L2)

# 二次代价函数
loss =tf.reduce_mean(tf.square(y-prediction))
# 使用梯度下降训练
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())#初始化数
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
        # 进行全部进行训练

    # 获得预测值
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
    # 只求最终的预测值
    # 画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()