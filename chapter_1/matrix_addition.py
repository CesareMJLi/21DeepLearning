#tensor flow matrix addition

import tensorflow as tf

# zeros=tf.Variable(tf.zeros([3, 3]))
# ones=tf.Variable(tf.ones([3]))
# result=zeros+ones
# print(result[0,:])

#define a variable to hold normal random values 
normal_rv = tf.Variable( tf.truncated_normal([2,3],stddev = 0.1))

new_v = tf.Variable(tf.ones(3))
#initialize the variable
init_op = tf.initialize_all_variables()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    print (sess.run(normal_rv))
    normal_rv=new_v+normal_rv
    print(sess.run(new_v))
    print (sess.run(normal_rv))

