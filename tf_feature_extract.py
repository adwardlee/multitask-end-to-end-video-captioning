import argparse
import os
import sys
import math
import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
from inception_resnet_v2 import *
import urllib2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
slim = tf.contrib.slim

model_path = './res_models'

video_path = '/data1/lijun/msvd/video-frames'

#video_train_feature_file = '/media/llj/storage/all_sentences/msvd_inception_globalpool_train_origin.txt'

#video_test_feature_file = '/media/llj/storage/all_sentences/msvd_inception_globalpool_test_origin.txt'

video_train_sent_file = '/data1/lijun/tensorflow_s2vt/msvd_sents_train_noval_lc_nopunc.txt'

video_test_sent_file = '/data1/lijun/tensorflow_s2vt/msvd_sents_test_lc_nopunc.txt'

vocabulary_file = '/data1/lijun/tensorflow_s2vt/msvd_vocabulary1.txt'

#model_name = '/home/llj/tensorflow_s2vt/multitask_models/initialize_with_two_model/10batch_size2alpha_0.01_multitask_model-72000'

model_name = '/data1/lijun/tensorflow_s2vt/new_attribute_models/5_lr_1e-05_batch_size16alpha_1attribute_model-14000'


dim_image = 1536
lstm_dim = 1000
word_dim = 500

n_lstm_step = 45
n_caption_lstm_step = 35
n_video_lstm_step = 10

n_epochs = 20
batch_size = 32
start_learning_rate = 0.01
width = 299
height = 299

def read_frames(video_path):
    vid = []
    for x in range(1301,1971):
        vid.append('vid'+str(x))
    image_dirs = {}
    for x in vid:
        image_dirs[x] = []
        #print('{},{}'.format(video_path, x))
        name = os.listdir(os.path.join(video_path,x))
        frame_cnt = len(name)
        #print('{}, {}'.format(x, frame_cnt))
        step = (frame_cnt-2) // 4
        frame_tick = range(1,min((2+ step * 4),frame_cnt),step)
        for tick in frame_tick:

            name = '{:06d}.jpg'.format(tick)
            frame = os.path.join(video_path,x, name)
            image_dirs[x].append(frame)
    return vid, image_dirs

def read_source_file(data_file):
	vid = []
	image_dirs = {}
	with open(data_file,'r') as files:
		for x in files:
			vid_name = x.strip().split(',')[0]
			if vid_name not in vid:
				vid.append(vid_name)
			if vid_name not in image_dirs:
				image_dirs[vid_name] = []
			image_dirs[vid_name].append(x.strip().split(',')[1])
	assert len(vid) == len(image_dirs)
	return vid,image_dirs

def image_reading_processing(path):
    video_batch = []
    for j in xrange(len(path)):
        #image_string = urllib2.read()
            #filename_queue = tf.train.string_input_producer([path[j]])
            #reader = tf.WholeFileReader()
            #key, value = reader.read(filename_queue)
            #with tf.gfile.FastGFile(path[j], 'rb') as f:
            #    image_data = f.read()
            #image = tf.image.decode_jpeg(image_data, channels=3)
            #if image.dtype != tf.float32:
            #    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            #image = tf.expand_dims(image, 0)
            #image = tf.image.resize_bilinear(image, [299,299], align_corners=False)
            #image = tf.subtract(image, 0.5)
            #image = tf.multiply(image, 2.0)
            image = cv2.imread(path[j], cv2.IMREAD_COLOR)### height,width,channels
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (299, 299))
            image = 2 * (image/255.0) - 1

            video_batch.append(image)
    return video_batch

def build_model():
    video_frames = tf.placeholder(tf.float32,[None, 299, 299, 3])
    all_frames = video_frames

    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2',
                               reuse=None) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=False):
                net, endpoints = inception_resnet_v2_base(all_frames, scope=scope) ###  8 x 8 x 1536
                net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope = 'AvgPool_1a_8x8')
                #net = slim.flatten(net)
    #video = tf.reshape(net,[-1, 98304])
    video = tf.reshape(net,[-1, 1536])
    #video = tf.squeeze(net)
    #video = tf.reshape(net, [-1, 1536])
    return video, video_frames

vid, image_dirs = read_frames(video_path)
#vid, image_dirs = read_source_file('/media/llj/storage/processed-data-ms/last8_frame_val.txt')
#vid, image_dirs = read_source_file('/media/llj/storage/processed-data-ms/all_video_list.txt')
#vid, image_dirs = read_source_file('/media/llj/storage/processed-data-ms/train_10frame_list.txt')
output_file = open('/data1/lijun/tensorflow_s2vt/att_5frame_test.txt','w')
features, frames = build_model()
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess = tf.InteractiveSession(config=config)
saver = tf.train.Saver(max_to_keep=100, write_version=1)
############## clipping every gradient####
#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss, global_step=global_step)
tf.global_variables_initializer().run()
#tf.summary.scalar('lr',learning_rate)
saver.restore(sess,model_name)
for index in vid:
    #output_file = open(out_file_name+'_'+str(index),'w')
    frame_num = 1
    one_video = image_reading_processing(image_dirs[index])
    #one_videos = sess.run(one_video)
    one_video_feature = []
    for fea_num in xrange(0, len(one_video), batch_size):
        input_frames = one_video[fea_num:fea_num+batch_size]
        out_feature = sess.run(features, feed_dict = {frames: input_frames})
        one_video_feature.extend(out_feature)
    one_video_feature = np.array(one_video_feature)
    for num in xrange(len(one_video_feature)):
        output_fea = one_video_feature[num,:]
        #output_file.write(' '.join(str(x) for x in output_fea.tolist()) + '\n')

	output_file.write(index + '_frame_' + str(frame_num) + ',' +\
							  ','.join(str(x) for x in output_fea.tolist()) + '\n')
        frame_num += 1


