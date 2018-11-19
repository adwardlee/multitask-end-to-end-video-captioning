import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
import cv2
import argparse
import matplotlib.pyplot as plt
import random
import math
from beam_search import *
from inception_resnet_v2 import *
import glob
from evaluation import *
import multiprocessing
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
slim = tf.contrib.slim

model_path = '/home/llj/tensorflow_s2vt/multilabel_models'

video_path = '/media/llj/storage/microsoft-corpus/youtube_frame_flow'

video_train_sent_file = '/media/llj/storage/all_sentences/msvd_sents_train_noval_lc_nopunc.txt'

video_realtest_sent_file = '/media/llj/storage/all_sentences/msvd_sents_test_lc_nopunc.txt'

video_test_sent_file = '/media/llj/storage/all_sentences/msvd_sents_val_lc_nopunc.txt'

vocab_file = '/home/llj/tensorflow_s2vt/train_most_freq_vocab_400_truncated.txt'

num_frame_per_video = 5
n_epochs = 100
batch_size = 8
start_learning_rate = 0.3
width = 299
height = 299
channels = 3
feature_dim = 1536
nums_label = 400
threshold = 0.5
conv_dim = 1000 ###################### choose 1000 or 500 #####

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Extract a CNN features')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=3, type=int)
    parser.add_argument('--net', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset to extract',
                        default='train_val', type=str)
    parser.add_argument('--task', dest='task',
                        help='train or test',
                        default='train', type=str)
    parser.add_argument('--tg', dest='tg',
                        help='target to be extract lstm feature',
                        default='/home/Hao/tik/jukin/data/h5py', type=str)
    parser.add_argument('--ft', dest='ft',
                        help='choose which feature type would be extract',
                        default='lstm1', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
			if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

class Multilabel():
    def __init__(self, batch_size, loss_weight = 1, decay_value = 0.00005, dropout_rate = 0.9,
                 width = 299, height = 299, channels= 3, feature_dim = 1536, label_dim=400):
        self.batch_size = batch_size
        self.loss_weight = loss_weight
        self.decay_value = decay_value
        self.dropout_rate = dropout_rate
        self.width = width
        self.height = height
        self.channels = channels
        self.label_dim = label_dim
        self.feature_dim = feature_dim
        self.n_video_lstm_step = num_frame_per_video

        self.attr_W = tf.Variable(tf.random_uniform([conv_dim, self.label_dim], -0.1, 0.1),dtype=tf.float32, name='attr_W', trainable=True)
        self.attr_b = tf.Variable(tf.zeros([self.label_dim], tf.float32),dtype=tf.float32, name='attr_b')


    def build_model(self):
        ########## inception resnet v2####
        ###preprocessing###
        loss = 0
        video_frames = tf.placeholder(tf.float32,[self.batch_size, self.n_video_lstm_step, self.height, self.width, self.channels])
        all_frames = tf.reshape(video_frames,[-1, self.height, self.width, self.channels])
        true_labels = tf.placeholder(tf.float32, [self.batch_size,self.label_dim])

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
          with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2',
                               reuse=None) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=False):
                net, endpoints = inception_resnet_v2_base(all_frames, scope=scope)
                net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope = 'AvgPool_1a_8x8')

                net = tf.stop_gradient(net)

                net = slim.fully_connected(net, conv_dim, activation_fn=None,scope = 'last_fc')
                #net = slim.conv2d(net,conv_dim,1,scope = 'attribute_conv1x1',normalizer_fn=None, normalizer_params=None)###### choose 1000, 500

                net = slim.flatten(net)
                net = slim.dropout(net, self.dropout_rate, is_training=True, scope='Dropout')

        video = tf.reshape(net, [self.batch_size, self.n_video_lstm_step, conv_dim])
        attribute_feature = tf.reduce_mean(video, axis=1)
        logits = tf.nn.xw_plus_b(attribute_feature, self.attr_W, self.attr_b)
        sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_labels,logits=logits)


        current_loss = (tf.reduce_sum(sigmoid_cross_entropy)/self.label_dim)/self.batch_size
        loss += self.loss_weight * current_loss #+ weight_decay_loss
                #for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                    #print(v)
        weight_decay_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' or 'BatchNorm' not in v.name]) \
                            * self.decay_value
        #loss = loss/tf.reduce_sum(caption_mask) + weight_decay_loss  ###normal loss
        loss = loss + weight_decay_loss  #### label smoothing
        return loss, video_frames, true_labels

    def evaluate_multilabel(self, threshold):

        video_frames = tf.placeholder(tf.float32,
                                  [None, self.n_video_lstm_step, self.height, self.width, self.channels])
        all_frames = tf.reshape(video_frames, [-1, self.height, self.width, self.channels])

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2',reuse = None) as scope:
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                    is_training = False):
                    #tf.get_variable_scope().reuse_variables()
                    net, endpoints = inception_resnet_v2_base(all_frames, scope=scope)
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')

                    net = slim.fully_connected(net, conv_dim, activation_fn=None,scope = 'last_fc')
                    #net = slim.conv2d(net,conv_dim,1,scope = 'attribute_conv1x1',normalizer_fn=None, normalizer_params=None)###### choose 1000, 500

                    net = slim.flatten(net)
                    net = slim.dropout(net, self.dropout_rate, is_training=False, scope='Dropout')
        video = tf.reshape(net, [-1, self.n_video_lstm_step, conv_dim])
        attribute_feature = tf.reduce_mean(video, axis=1)
        logits = tf.nn.xw_plus_b(attribute_feature, self.attr_W, self.attr_b)
        scores = tf.sigmoid(logits)#### batch_size x label_num

        return video_frames, scores

def get_metrics(scores, labels, num_videos):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    count_pos = 0
    count_neg = 0
    for i in xrange(num_videos):
        for j in xrange(nums_label):
            if labels[i][j] >= threshold:
                true_positive += (scores[i][j] >= threshold)
                false_negative += (scores[i][j] < threshold)
                count_pos += 1
            if labels[i][j] < threshold:
                true_negative += (scores[i][j] < threshold)
                false_positive += (scores[i][j] >= threshold)
                count_neg += 1
    return true_positive, true_negative, false_positive, false_negative, count_pos, count_neg

def get_video_frame_path(sent_file, frame_path=video_path, num_frame_per_video = num_frame_per_video, prefix='frame_'):
    sents = {}
    vid = []
    video_frames = {}
    with open(sent_file, 'r') as video_sent_file:
        for line in video_sent_file:
            line = line.strip()
            id_sent = line.split('\t')
            if id_sent[0] not in sents:
                sents[id_sent[0]] = []
            sents[id_sent[0]].append(id_sent[1])
            if id_sent[0] not in vid:
                vid.append(id_sent[0])
    for vid_name in vid:
        video_frames[vid_name] = []
        video_path = frame_path + '/' + vid_name
        frame_cnt = len(glob.glob(video_path+'/'+prefix+'*'))
        step = (frame_cnt-1)//(num_frame_per_video-1)
        if step >0 :
            frame_ticks = range(1, min((2 + step * (num_frame_per_video-1)), frame_cnt+1), step)
        else:
            frame_ticks = [1]*num_frame_per_video
        for tick in frame_ticks:
            name = '{}{:06d}.jpg'.format(prefix, tick)
            frame = os.path.join(video_path,name)
            video_frames[vid_name].append(frame)
            #frame = cv2.resize(frame,(340,256)) ### width,height

    feature_length = [len(v) for v in video_frames.values()]
    print 'length: ', set(feature_length)
    assert len(set(feature_length)) == 1  ######## make sure the feature lengths are all the same
    return sents, video_frames, vid

def image_reading_processing(path):
    video_batch = [[] for x in xrange(len(path))]
    for i in xrange(len(path)):
        for j in xrange(num_frame_per_video):
            image = cv2.imread(path[i][j], cv2.IMREAD_COLOR)### height,width,channels
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
            image = image.astype(np.float32)
            image = 2 * (image/255.0) - 1

            video_batch[i].append(image)
    return video_batch

def read_sent_vocab_file(sent_file, vocab_file):
    label_num = 0
    vid_sent = dict()
    vocab = list()
    with open(sent_file, 'r') as f:
        for line in f:
            line = line.strip()
            id_sent = line.split('\t')
            if id_sent[0] not in vid_sent:
                vid_sent[id_sent[0]] = []
            vid_sent[id_sent[0]].append(id_sent[1])

    with open(vocab_file, 'r') as f:
        for line in f:
            line = line.strip()
            vocab.append(line)
            label_num += 1
    return vid_sent, vocab, label_num


def get_multilabel(vid_sentence, vocabulary):
    vid_label = defaultdict()
    for vid, sents in vid_sentence.iteritems():
        label = list()
        for x in sents:
            label1 = list()
            for index, v in enumerate(vocabulary):
                count = 0
                for a in x.split():
                    if v == a:
                        count = 1
                label1.extend([count])
            label.append(label1)
            # print 'label1: ',label1
        label = np.sum(np.array(label), axis=0)
        np.putmask(label, label > 0, 1)

        # print 'label: ',label
        vid_label[vid] = label
    return vid_label

def train():
    vid_sent, vocab, label_num = read_sent_vocab_file(video_train_sent_file, vocab_file)
    assert nums_label == label_num, 'vocab file label number is not equal to nums_label'
    train_captions, train_video_frames, train_vid = get_video_frame_path(sent_file=video_train_sent_file ,frame_path=video_path, num_frame_per_video=num_frame_per_video)
    test_captions, test_video_frames, test_vid = get_video_frame_path(sent_file=video_test_sent_file,frame_path=video_path, num_frame_per_video=num_frame_per_video)
    train_labels = get_multilabel(train_captions, vocab)
    test_labels = get_multilabel(test_captions, vocab)
    model = Multilabel(batch_size, loss_weight=1, decay_value=0.00005, dropout_rate=0.9,
        width=width, height=height, channels=channels, feature_dim=feature_dim, label_dim=label_num)

    tf_loss, tf_video_frames, tf_labels = model.build_model()
    tf_test_video_frames, tf_scores = model.evaluate_multilabel(threshold=threshold)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    # config.gpu_options.allocator_type = 'BFC'
    sess = tf.InteractiveSession(config=config)
    saver = tf.train.Saver(max_to_keep=100, write_version=1)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                               6000, 0.5, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    ###################clipping every gradient ################
    gvs = optimizer.compute_gradients(tf_loss, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    capped_gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    ############## clipping every gradient####
    tf.global_variables_initializer().run()
    optimistic_restore(sess, 'inception_resnet_v2_2016_08_30.ckpt')

    loss_to_draw = []

    for epoch in xrange(0,n_epochs):
        loss_to_draw_epoch = []
        index = list(xrange(len(train_vid)))
        random.shuffle(index)
        for start, end in zip(range(0, len(index) - batch_size, batch_size), range(batch_size, len(index), batch_size)):
            start_time = time.time()
            id = [train_vid[x] for x in index[start:end]]
            samples = [train_video_frames[x] for x in id]
            sample_labels = [train_labels[x] for x in id]
            video_batch = image_reading_processing(samples)

            _, loss_value =sess.run([train_op,tf_loss],feed_dict={tf_video_frames: video_batch, tf_labels: sample_labels})
            loss_to_draw_epoch.append(loss_value)

            print 'idx: ', start, ' rate: ', sess.run(learning_rate), " Epoch: ", epoch, " loss: ", loss_value, \
                ' Elapsed time: ', str((time.time() - start_time))

        with open('./multilabel_models/evaluate_multilabel_fcconv3e-1_stopgradient_dim1000_val.txt','a') as f:
            scores = []
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0
            count_pos = 0
            count_neg = 0
            mAP=0
            AUC=0
            batch_num = 0
            for aa in xrange(0,len(test_vid),batch_size):
                id = test_vid[aa:aa+batch_size]
                num_videos = len(id)
                test_sample_labels = [test_labels[x] for x in id]
                test_samples = [test_video_frames[x] for x in id]
                test_video_batch = image_reading_processing(test_samples)

                scores = sess.run(tf_scores, feed_dict={tf_test_video_frames: test_video_batch})
                temp_true_positive, temp_true_negative, temp_false_positive, temp_false_negative, temp_count_pos,temp_count_neg=get_metrics(scores,
                                                                                        test_sample_labels, num_videos)
                true_positive += temp_true_positive
                true_negative += temp_true_negative
                false_positive += temp_false_positive
                false_negative += temp_false_negative
                count_pos += temp_count_pos
                count_neg += temp_count_neg

                ##### map
                #test_sample_labels = np.array(test_sample_labels)
                for j in xrange(num_videos):
                    mAP+=average_precision_score(test_sample_labels[j], scores[j])
                #### AUC
                for j in xrange(num_videos):
                    AUC+=roc_auc_score(test_sample_labels[j],scores[j])


            sensitivity = true_positive / float(count_pos) if count_pos > 0 else 0
            specificity = true_negative / float(count_neg) if count_neg > 0 else 0
            harmmean = 2.0 / (count_pos / float(true_positive) + count_neg / float(true_negative)) if ((count_pos + count_neg) > 0) else 0
            precision = (true_positive / float(true_positive + false_positive)) if (true_positive > 0) else 0
            f1_score = 2.0 * true_positive / float(2 * true_positive + false_positive + false_negative) if (true_positive > 0) else 0

            ### map
            mAP = mAP/float(len(test_vid))
            ##AUC
            AUC = AUC/float(len(test_vid))

            f.write('\n')
            f.write('Epoch %d\n' % epoch)
            f.write('\n')
            f.write("sensitivity:" + str(sensitivity))
            f.write('\n')
            f.write("specificity:" + str(specificity))
            f.write('\n')
            f.write("harmmean:" + str(harmmean))
            f.write('\n')
            f.write("precision:" + str(precision))
            f.write('\n')
            f.write("f1_score:" + str(f1_score))
            f.write('\n')
            f.write('mAP: ' + str(mAP))
            f.write('\n')
            f.write('AUC: '+ str(AUC))
            f.write('\n')

        # with open('./multilabel_models/evaluate_multilabel_1x1conv1e-3_stopgradient_dim500_test.txt','a') as f:
        #     scores = []
        #     true_positive = 0
        #     true_negative = 0
        #     false_positive = 0
        #     false_negative = 0
        #     count_pos = 0
        #     count_neg = 0
        #     mAP=0
        #     AUC=0
        #     batch_num = 0
        #     for aa in xrange(0,len(realtest_vid),batch_size):
        #         id = realtest_vid[aa:aa+batch_size]
        #         num_videos = len(id)
        #         test_sample_labels = [realtest_labels[x] for x in id]
        #         test_samples = [realtest_video_frames[x] for x in id]
        #         test_video_batch = image_reading_processing(test_samples)
        #
        #         scores = sess.run(tf_scores, feed_dict={tf_test_video_frames: test_video_batch})
        #         temp_true_positive, temp_true_negative, temp_false_positive, temp_false_negative, temp_count_pos,temp_count_neg=get_metrics(scores,
        #                                                                                 test_sample_labels, num_videos)
        #         true_positive += temp_true_positive
        #         true_negative += temp_true_negative
        #         false_positive += temp_false_positive
        #         false_negative += temp_false_negative
        #         count_pos += temp_count_pos
        #         count_neg += temp_count_neg
        #
        #         ##### map
        #         #test_sample_labels = np.array(test_sample_labels)
        #         for j in xrange(num_videos):
        #             mAP+=average_precision_score(test_sample_labels[j], scores[j])
        #         #### AUC
        #         for j in xrange(num_videos):
        #             AUC+=roc_auc_score(test_sample_labels[j],scores[j])
        #
        #
        #     sensitivity = true_positive / float(count_pos) if count_pos > 0 else 0
        #     specificity = true_negative / float(count_neg) if count_neg > 0 else 0
        #     harmmean = 2.0 / (count_pos / float(true_positive) + count_neg / float(true_negative)) if ((count_pos + count_neg) > 0) else 0
        #     precision = (true_positive / float(true_positive + false_positive)) if (true_positive > 0) else 0
        #     f1_score = 2.0 * true_positive / float(2 * true_positive + false_positive + false_negative) if (true_positive > 0) else 0
        #
        #     ### map
        #     mAP = mAP/float(len(test_vid))
        #     ##AUC
        #     AUC = AUC/float(len(test_vid))
        #
        #     f.write('\n')
        #     f.write('Epoch %d\n' % epoch)
        #     f.write('\n')
        #     f.write("sensitivity:" + str(sensitivity))
        #     f.write('\n')
        #     f.write("specificity:" + str(specificity))
        #     f.write('\n')
        #     f.write("harmmean:" + str(harmmean))
        #     f.write('\n')
        #     f.write("precision:" + str(precision))
        #     f.write('\n')
        #     f.write("f1_score:" + str(f1_score))
        #     f.write('\n')
        #     f.write('mAP: ' + str(mAP))
        #     f.write('\n')
        #     f.write('AUC: '+ str(AUC))
        #     f.write('\n')


        loss_to_draw.append(np.mean(loss_to_draw_epoch))
        plt_save_dir = "./multilabel_loss_imgs"
        plt_save_img_name = str(epoch) + '_1e-3_stopgradient.png'
        plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
        plt.grid(True)
        plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))

        if np.mod(epoch, 2) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'batch_size' + str(batch_size)+ 'multilabel_fc3e-1_stopgradient_dim1000'), global_step=epoch)


def evaluation():
    vid_sent, vocab, label_num = read_sent_vocab_file(video_train_sent_file, vocab_file)
    assert nums_label == label_num, 'vocab file label number is not equal to nums_label'

    test_captions, test_video_frames, test_vid = get_video_frame_path(video_realtest_sent_file,frame_path=video_path, num_frame_per_video=num_frame_per_video)
    test_labels = get_multilabel(test_captions, vocab)
    model = Multilabel(batch_size, loss_weight=1, decay_value=0.00005, dropout_rate=1,
            width=width, height=height, channels=channels, feature_dim=feature_dim, label_dim=label_num)

    tf_test_video_frames, tf_scores = model.evaluate_multilabel(threshold=threshold)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.InteractiveSession(config=config)
    saver = tf.train.Saver()

    with open('./multilabel_models/evaluate_multilabel_fcconv3e-1_stopgradient_dim1000_test.txt','a') as f:
      for i in xrange(34,36,4):
        model_path_last = model_path + '/batch_size8multilabel_fc3e-1_stopgradient_dim1000-' + str(i)
        saver.restore(sess,model_path_last)
        scores = []
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        count_pos = 0
        count_neg = 0
        mAP=0
        AUC=0
        for aa in xrange(0,len(test_vid),batch_size):
            id = test_vid[aa:aa+batch_size]
            num_videos = len(id)
            test_sample_labels = [test_labels[x] for x in id]
            test_samples = [test_video_frames[x] for x in id]
            test_video_batch = image_reading_processing(test_samples)

            scores = sess.run(tf_scores, feed_dict={tf_test_video_frames: test_video_batch})
            temp_true_positive, temp_true_negative, temp_false_positive, temp_false_negative, temp_count_pos,temp_count_neg=get_metrics(scores,
                                                                                    test_sample_labels, num_videos)
            true_positive += temp_true_positive
            true_negative += temp_true_negative
            false_positive += temp_false_positive
            false_negative += temp_false_negative
            count_pos += temp_count_pos
            count_neg += temp_count_neg

            ##### map
            scores = np.array(scores)
            test_sample_labels = np.array(test_sample_labels)
            for j in xrange(num_videos):
                mAP+=average_precision_score(test_sample_labels[j], scores[j])
            #### AUC
            for j in xrange(num_videos):
                AUC+=roc_auc_score(test_sample_labels[j],scores[j])


        sensitivity = true_positive / float(count_pos) if count_pos > 0 else 0
        specificity = true_negative / float(count_neg) if count_neg > 0 else 0
        harmmean = 2.0 / (count_pos / float(true_positive) + count_neg / float(true_negative)) if ((count_pos + count_neg) > 0) else 0
        precision = (true_positive / float(true_positive + false_positive)) if (true_positive > 0) else 0
        f1_score = 2.0 * true_positive / float(2 * true_positive + false_positive + false_negative) if (true_positive > 0) else 0

        ### map
        mAP = mAP/float(len(test_vid))
        ##AUC
        AUC = AUC/float(len(test_vid))

        f.write('\n')
        f.write('Epoch %d\n' % i)
        f.write('\n')
        f.write("sensitivity:" + str(sensitivity))
        f.write('\n')
        f.write("specificity:" + str(specificity))
        f.write('\n')
        f.write("harmmean:" + str(harmmean))
        f.write('\n')
        f.write("precision:" + str(precision))
        f.write('\n')
        f.write("f1_score:" + str(f1_score))
        f.write('\n')
        f.write('mAP: ' + str(mAP))
        f.write('\n')
        f.write('AUC: '+ str(AUC))
        f.write('\n')



if __name__ == '__main__':
    args = parse_args()
    if args.task == 'train':
        with tf.device('/gpu:' + str(args.gpu_id)):
            train()
    elif args.task == 'evaluate':
        with tf.device('/gpu:' + str(args.gpu_id)):
            evaluation()
