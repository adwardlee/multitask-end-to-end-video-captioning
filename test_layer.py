import tensorflow as tf
import numpy as np

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.framework import tensor_shape


class Fisher_pooling(base._Layer):


    def __init__(self,normalization=True,GMM_number=2,
                 log_likelihood=False,input_dimension = 6, spatial_dim = 5, trainable= True,name=None,
               **kwargs):
	super(Fisher_pooling, self).__init__(trainable=trainable,
                                name=name, **kwargs)
        self.normalization=True
        self.GMM_number=GMM_number
        self.log_likelihood=log_likelihood
     
        self.sample_dimension=input_dimension
	self.spatial_dim = spatial_dim
	
    def build(self, input_shape):            
	#input_shape = tensor_shape.Tensorshape(input_shape)
        self.GMM_mean = tf.Variable(tf.random_uniform([self.GMM_number, self.sample_dimension],-0.05,0.05),dtype=tf.float32,name = 'fisher_mean', trainable=True)
        self.GMM_beta = tf.Variable(tf.random_uniform([self.GMM_number, self.sample_dimension],-0.05,0.05),dtype=tf.float32,name = 'fisher_beta', trainable=True)
        self.GMM_alpha = tf.Variable(tf.ones([self.GMM_number],tf.float32),dtype=tf.float32,name = 'fisher_alpha', trainable=True)

    def call(self,inputs):
	X= inputs
	print 'shape: ',X.get_shape()
        #X = tf.placeholder(tf.float32,[None, self.spatial_dim, self.spatial_dim, self.sample_dimension])
        X_permute = tf.reshape(X,[self.sample_dimension,-1]) #Flatten acitvations. reshaped_X.shape=[num_samepls,rows*cols,num_filters]
	output1 = 0.1*X_permute#tf.matmul(self.GMM_mean,X_permute)
	return output1

ff = Fisher_pooling()

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# config.gpu_options.allocator_type = 'BFC'

a = tf.placeholder(tf.float32, [None,9])

c = ff(a)
sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)

dd = sess.run(c, feed_dict={a:np.ones((4,9))})
print 'dd: ',dd
