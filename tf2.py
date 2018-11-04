"""实现了两个矩阵的相乘操作"""
# __author__ = 'HAPPYYOUNG'
# import tensorflow as tf
# matrix1 = tf.constant([[1,2]])
# matrix2 = tf.constant([[3],[4]])
# product = tf.matmul(matrix1,matrix2)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(product))

#
# """实现了某个变量的自加操作"""
# import tensorflow as tf
# count = tf.Variable([0],name='counter')
# one = tf.constant(1,name='one')
#
# new_value = tf.add(count,one)
# update = tf.assign(count,new_value)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for _ in range(4):
#     sess.run(update)
#     print(sess.run(count))

# """placeholder的使用"""
# import tensorflow as tf
# input1 = tf.placeholder(tf.float32, [2,3], name='input1')
# input2 = tf.placeholder(tf.float32, [3,2], name='input2')
# product = tf.matmul(input1,input2)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(product,feed_dict={input1:[[1,2,3],[4,5,6]],input2:[[1,2],[3,4],[5,6]]}))

"""动画展现迭代拟合曲线的过程"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros[1,out_size]+0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
    if activation_function == None:
        pass
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.show(block=False)

with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction,feed_dict={xs:x_data})
            lines = ax.plot(x_data,prediction_value,'r-',lw=5)
            plt.pause(0.1)