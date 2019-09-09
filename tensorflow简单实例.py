import tensorflow as tf
import numpy as np

x_data=np.random.random(100)
y_data=x_data*0.1+0.2

# 构造一个线性模型
b=tf.Variable(0.)
k=tf.Variable(0.)
y=k*x_data+b

# 二次代价函数
loss=tf.reduce_mean(tf.square(y_data-y))

#定义一个梯度下降法来进行训练的优化器
optimizer=tf.train.GradientDescentOptimizer(0.02)

# 最小化代价函数
train=optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(300):
        sess.run(train)
        if step%20 ==0:
            print(step,sess.run([k,b]))