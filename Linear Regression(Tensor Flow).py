# machine learning for linear regression
import tensorflow as tf

w1 =tf.Variable([-.3],tf.float32)
w2 =tf.Variable([-.6],tf.float32)
b =tf.Variable([.3],tf.float32)

x= tf.placeholder(tf.float32)
linear_model=  w1*x+w2*(x**2)+b

y=tf.placeholder(tf.float32)

squared_delta=tf.square(linear_model-y)

loss=tf.reduce_sum(squared_delta)

init=tf.global_variables_initializer()

#optimizer

optimizer =tf.train.GradientDescentOptimizer(0.001)
train= optimizer.minimize(loss)


sess=tf.Session()

sess.run(init)

for i in range(1000):

 sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-6]})
 print(sess.run([w1,w2,b]))
#print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
