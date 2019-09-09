import tensorflow as tf

x=tf.Variable([1,2])#这个变量的使用需要进行初始化
a=tf.constant([3,3])
#增加一个减法的op
sub=tf.subtract(x,a)
#增加一个加法的op
add=tf.add(x,sub)
init =tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

#创建一个变量初始化为0
state= tf.Variable(0,name='counter')
new_value=tf.add(state,1)
update=tf.assign(state,new_value)#讲后面的值赋值给前面值
init2=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init2)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))