import tensorflow as tf
import numpy as np
import pickle
import multiprocessing
import os 
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

step = 5
segment = 3
frame_ticks = []
b = random.sample(xrange(step),1)
print 'b: ',int(b[0])
frame_ticks.extend([1 + int(x) + segment * step for x in random.sample(xrange(step), 3)])
print(frame_ticks)
for x in frame_ticks:
  print 'x: ',x

bb = (1,2)
a = tuple(b for b in bb)
c,d = a
print 'a: ',c,d
print 'cpu: ',multiprocessing.cpu_count()

aaa = 15%5//2
print 'aaa: ',aaa
a = np.array([1.2,3.2,2.4])
print 'a: ', a[:5]
sess = tf.Session()


vector = tf.convert_to_tensor([[[1,3],[4,6],[7,9]],[[2,4],[6,8],[10,12]]])
print 'shape: ', sess.run(tf.shape(vector))
print 'vector: ', sess.run(tf.reduce_mean(vector,axis=1))
print 'vector shape: ', sess.run(tf.shape(tf.reduce_mean(vector,axis=1)))

captions = tf.convert_to_tensor([[1,2],[3,4],[5,0]])
print 'captions type: ',captions.dtype
mask = tf.to_float(tf.not_equal(captions, tf.convert_to_tensor([0])))
print 'mask: ',sess.run(mask)

temp = tf.convert_to_tensor([[1],[2],[3]],dtype=tf.float32)
temp1 = tf.reshape(temp,[-1])
print 'temp: ',sess.run(tf.shape(temp))
print 'temp1: ',sess.run(tf.shape(temp1))

one_mask = [[1] for x in xrange(16)]
one_mask = tf.Variable(one_mask,trainable=False,dtype=tf.float32)
print 'one mask shape: ',sess.run(tf.shape(one_mask))

k = tf.convert_to_tensor([600],dtype=tf.float64)
onehundred_percent = tf.convert_to_tensor([1.00001], dtype=tf.float64)
step_num = tf.Variable(200,trainable=False,dtype=tf.float64)
sess.run(tf.global_variables_initializer())
true_word_prob = tf.expand_dims(tf.divide(k,tf.add(k,tf.exp(tf.divide(step_num,k)))),0)
pre_prob = tf.concat([true_word_prob,tf.subtract(onehundred_percent,true_word_prob)],1)
probabilities =tf.tile(pre_prob,[8,1])
log_probs = tf.log(probabilities)
indice = tf.multinomial(log_probs, num_samples=1)
print 'indice shape: ',sess.run(tf.shape(indice))
print 'indice: ',sess.run(indice)
print 'log_probs: ',sess.run(log_probs)

input1 = tf.convert_to_tensor([[40965],[40964],[40963],[40962],[40961]])
input2 = tf.convert_to_tensor([[50000],[50001],[50002],[50003],[50004]])
abc = []
print 'shape : ',sess.run(tf.shape(input1))
all_inputs = tf.concat([input1,input2],1)
all_inputs1 = tf.concat([input1,input2],1)
print 'all inputs: ',sess.run(all_inputs)
batch_tensors = all_inputs
probabilities = tf.tile(tf.convert_to_tensor([[0.9,0.1]]),[5,1])
print 'probs : ', sess.run(probabilities)
rescaled_probas = tf.log(probabilities)  # shape [1, batch_size]
print 'rescaled_probas : ', sess.run(rescaled_probas)
# We can now draw one example from the distribution (we could draw more)
a = tf.cast(tf.expand_dims(tf.range(0,5),1),tf.int64)
for i in xrange(8):
  indice = tf.multinomial(rescaled_probas, num_samples=1)
  indice1 = tf.concat([a,indice],1)
  abc.append(tf.gather_nd(batch_tensors,indice1))
print 'concat : ', sess.run(abc)
output = tf.gather_nd(batch_tensors,indice1)
output1 = tf.gather_nd(all_inputs1,indice1)
output_all = tf.concat([output,output1],0)
print 'all output is : ',sess.run(output_all)
print 'output is : ',sess.run(output)
print 'output shape: ',sess.run(tf.shape(output))

