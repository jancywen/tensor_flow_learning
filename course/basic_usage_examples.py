# -*- coding: utf-8 -*-


import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" 使用默认图(default graph) """
# 1, 构建计算图
matrix1 = tf.constant([[3., 3.]])

matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix1, matrix2)

# 2, 在会话中载入图,执行图,执行完毕后关闭
sess = tf.Session()

# result = sess.run(product)
#
# print(result)
#
# sess.close()


with tf.Session() as sess:
    result = sess.run([product])
    # print(result)


""" 使用交互式 """

sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

x.initializer.run()
sub = tf.subtract(x, a)
print(sub.eval())
sess.close()


""" 用变量构建一个简单的计数器 """

state = tf.Variable(0, name='counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
# 在启动图形之后，必须通过运行“init”操作对变量进行初始化
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))

    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


""" fetches 多个tensor值 """

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(4.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)


""" Feed """
"""
feed 使用一个 tensor 值临时替换一个操作的输出结果. 你可以提供 feed 数据作为
run() 调用的参数.feed 只在调用它的方法内有效, 方法结束, feed 就会消失. 最常见的用
例是将某些特殊的操作指定为"feed" 操作, 标记的方法是使用tf.placeholder()为这些操
作创建占位符.
"""
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1: [7.0], input2: [2.0]})
    print(result)
