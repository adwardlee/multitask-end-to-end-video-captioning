import matplotlib
matplotlib.use('Agg')

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
from rouge_evaluation import *
#from evaluation import *
import multiprocessing
from collections import defaultdict
import operator
import psutil

slim = tf.contrib.slim

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

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, word_dim, lstm_dim, batch_size, n_lstm_steps, n_video_lstm_step,
                 n_caption_lstm_step, bias_init_vector=None, loss_weight = 1, decay_value = 0.00005, dropout_rate = 0.9,
                 width = 299, height = 299, channels= 3, feature_dim = 1536, label_dim=400, alpha = 0.2):
        self.dim_image = dim_image
        self.n_words = n_words
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps  #### number of lstm cell
        self.n_video_lstm_step = n_video_lstm_step  ### frame number
        self.n_caption_lstm_step = n_caption_lstm_step  #### caption number
        self.loss_weight = loss_weight
        self.decay_value = decay_value
        self.dropout_rate = dropout_rate
        self.width = width
        self.height = height
        self.channels = channels
        self.label_dim = label_dim
        self.feature_dim = feature_dim
        self.alpha = alpha

        with tf.device("/cpu:0"):
            self.Wemb = self.Wemb = tf.Variable(tf.random_uniform([n_words, word_dim], -0.1, 0.1,seed=1), dtype=tf.float32,
                                                name='Wemb',trainable=True)  ##without cpu

        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=False)
        self.lstm1_dropout = tf.contrib.rnn.DropoutWrapper(self.lstm1, output_keep_prob=self.dropout_rate)
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=False)
        self.lstm2_dropout = tf.contrib.rnn.DropoutWrapper(self.lstm2, output_keep_prob=self.dropout_rate)

        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, word_dim], -0.1, 0.1,seed=1), dtype=tf.float32,
                                          name='encode_image_W', trainable=True)
        self.encode_image_b = tf.Variable(tf.zeros([word_dim], tf.float32), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([lstm_dim, n_words], -0.1, 0.1,seed=1), dtype=tf.float32,
                                        name='embed_word_W', trainable=True)
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')
        #### multilabel fc layer weights and bias
        self.attr_W = tf.Variable(tf.random_uniform([self.feature_dim, self.label_dim], -0.1, 0.1,seed=1), dtype=tf.float32,
                                  name='attr_W', trainable=True)
        self.attr_b = tf.Variable(tf.zeros([self.label_dim], tf.float32), dtype=tf.float32, name='attr_b')

    def build_model(self):
        ########## inception resnet v2####
        ###preprocessing###
        video_frames = tf.placeholder(tf.float32,[self.batch_size, self.n_video_lstm_step, self.height, self.width, self.channels])
        all_frames = tf.reshape(video_frames,[-1, self.height, self.width, self.channels])
        true_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim])

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
          with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2',
                               reuse=None) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=False):
                net, endpoints = inception_resnet_v2_base(all_frames, scope=scope)
                net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope = 'AvgPool_1a_8x8')
                net = slim.flatten(net)
                net = slim.dropout(net, self.dropout_rate, is_training=True, scope='Dropout')
        video = tf.reshape(net, [self.batch_size, self.n_video_lstm_step, self.dim_image])

        #####################################################


        #video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])  ###llj

        caption = tf.placeholder(tf.int32, [self.batch_size,
                                            self.n_caption_lstm_step])  ####llj    make caption start at n_video_lstm_step
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step])  ##llj

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W,
                                    self.encode_image_b)  # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_lstm_step,
                                           self.word_dim])  ########potential problem in reshape
        ### add dropout####
        image_emb = tf.layers.dropout(inputs=image_emb,rate=self.dropout_rate)

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size], tf.float32)
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size], tf.float32)
        padding = tf.zeros([self.batch_size, self.word_dim], tf.float32)

        probs = []
        loss = 0.0


        ##############################  Encoding Stage ##################################
        with tf.variable_scope("s2vt") as scope:
            for i in range(0, self.n_video_lstm_step):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1_dropout(image_emb[:, i, :], state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2_dropout(tf.concat([output1, padding], 1), state2)

                    ############################# Decoding Stage ######################################

            for i in range(0, self.n_caption_lstm_step):  ## Phase 2 => only generate captions
                if i == 0:
                    with tf.device("/cpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([self.batch_size],
                                                                                  dtype=tf.int64))  ######## embedding begin of sentence <bos>
                else:

                    with tf.device("/cpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i - 1])  ##without cpu    ### i-1 correspond to the previous word
                #### add dropout##
                current_embed = tf.layers.dropout(inputs=current_embed,rate = self.dropout_rate)

                tf.get_variable_scope().reuse_variables()
                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1_dropout(padding, state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2_dropout(tf.concat([output1, current_embed], 1), state2)

                    # labels = tf.expand_dims(caption[:, i], 1)#### batch_size x 1 ####### i correspond to current word
                    # indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) ###### batch_size x 1
                    # concated = tf.concat([indices, labels],1)  #### make indices and labels pair batchsize x 2
                # onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words],axis=0), 1.0, 0.0) ##### batch_size number of one hot word vector### batch_size x n_words

                labels = tf.convert_to_tensor(caption[:, i])  #### batch_size x 1 ####### i correspond to current word
                onehot_labels = tf.one_hot(labels, self.n_words, 1.0, 0.0)

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_words) ### normal losss
                cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logit_words, label_smoothing=0.05) ##### label smoothing
                cross_entropy = cross_entropy * caption_mask[:,i]  ######### need to move caption_mask (cont_sent)one column left ##########################llj
                probs.append(logit_words)

                current_loss = tf.reduce_sum(cross_entropy)
                loss += self.loss_weight * current_loss #+ weight_decay_loss
                #for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                    #print(v)

        ######## multilabel loss########
        attribute_feature = tf.reduce_mean(video, axis=1)
        logits = tf.nn.xw_plus_b(attribute_feature, self.attr_W, self.attr_b)
        sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_labels, logits=logits)

        ##### divide the batch_size * label_num ####
        #sigmoid_cross_entropy = sigmoid_cross_entropy/(self.label_dim*1.0*self.batch_size)

        #multilabel_loss = self.loss_weight * tf.reduce_sum(sigmoid_cross_entropy)
        ########## divide the batch_size * label_num ###
        multilabel_loss = tf.reduce_sum(sigmoid_cross_entropy)
        weight_decay_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' or 'BatchNorm' not in v.name]) \
                            * self.decay_value
        #loss = loss/tf.reduce_sum(caption_mask) + weight_decay_loss  ###normal loss
        loss = (1-alpha) * loss / tf.reduce_sum(caption_mask) + weight_decay_loss + alpha * multilabel_loss
        return loss, video_frames, caption, caption_mask, probs, true_labels

    def build_generator(self, beam_size=1, length_normalization_factor=0.5):
        video_frames = tf.placeholder(tf.float32,
                                      [1, self.n_video_lstm_step, self.height, self.width, self.channels])
        all_frames = tf.reshape(video_frames, [-1, self.height, self.width, self.channels])

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2',
                                   reuse=None) as scope:
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=False):
                    tf.get_variable_scope().reuse_variables()
                    net, endpoints = inception_resnet_v2_base(all_frames, scope=scope)
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
                    net = slim.dropout(net, self.dropout_rate, is_training=False, scope='Dropout')
        video = tf.reshape(net, [1, self.n_video_lstm_step, self.dim_image])

        #video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.word_dim])
        state1 = tf.zeros([1, self.lstm1.state_size], tf.float32)
        state2 = tf.zeros([1, self.lstm2.state_size], tf.float32)
        padding = tf.zeros([1, self.word_dim], tf.float32)

        sentence = []
        probs = []
        with tf.variable_scope("s2vt") as scope:
            for i in range(0, self.n_video_lstm_step):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(image_emb[:, i, :], state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([output1, padding], 1), state2)

                    ############### decoding ##########
                    # with tf.variable_scope("s2vt") as scope:
            for i in range(0, self.n_caption_lstm_step):
                tf.get_variable_scope().reuse_variables()
                if i ==0:
                    with tf.device('/cpu:0'):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1],dtype=tf.int64))
                else:
                    with tf.device('/cpu:0'):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                        current_embed = tf.expand_dims(current_embed, 0)

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding, state1)
                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([output1, current_embed], 1), state2)
                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                words_probabilities = tf.exp(logit_words) / tf.reduce_sum(tf.exp(logit_words), -1)
                max_prob_index = tf.argmax(words_probabilities, 1)[0]
                sentence.append(max_prob_index)
               # probs.append(words_probabilities[max_prob_index])
                if max_prob_index == 0:
                    break
        return video_frames, sentence, probs

    def build_loss(self):
        ########## inception resnet v2####
        ###preprocessing###
        video_frames = tf.placeholder(tf.float32,
                                      [self.batch_size, self.n_video_lstm_step, self.height, self.width, self.channels])
        all_frames = tf.reshape(video_frames, [-1, self.height, self.width, self.channels])
        true_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim])

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2',
                                   reuse=None) as scope:
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=False):
                    tf.get_variable_scope().reuse_variables()
                    net, endpoints = inception_resnet_v2_base(all_frames, scope=scope)
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
                    net = slim.dropout(net, self.dropout_rate, is_training=True, scope='Dropout')
        video = tf.reshape(net, [self.batch_size, self.n_video_lstm_step, self.dim_image])

        #####################################################

        caption = tf.placeholder(tf.int32, [self.batch_size,
                                            self.n_caption_lstm_step])  ####llj    make caption start at n_video_lstm_step
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step])  ##llj
        # caption_mask = tf.to_float(tf.not_equal(caption,tf.convert_to_tensor([0]))) #### define mask in function
        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W,
                                    self.encode_image_b)  # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_lstm_step,
                                           self.word_dim])  ########potential problem in reshape
        ### add dropout####
        #image_emb = tf.layers.dropout(inputs=image_emb, rate=self.dropout_rate)

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size], tf.float32)
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size], tf.float32)
        padding = tf.zeros([self.batch_size, self.word_dim], tf.float32)

        logprobs = []
        loss = []

        ##############################  Encoding Stage ##################################
        with tf.variable_scope("s2vt") as scope:
            for i in range(0, self.n_video_lstm_step):
                #if i > 0:
                tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1_dropout(image_emb[:, i, :], state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2_dropout(tf.concat([output1, padding], 1), state2)

                    ############################# Decoding Stage ######################################

            for i in range(0, self.n_caption_lstm_step):  ## Phase 2 => only generate captions
                if i == 0:
                    with tf.device("/cpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([self.batch_size],
                                                                                  dtype=tf.int64))  ######## embedding begin of sentence <bos>
                else:

                    with tf.device("/cpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,
                                                                          i - 1])  ##without cpu    ### i-1 correspond to the previous word
                #### add dropout##
                #current_embed = tf.layers.dropout(inputs=current_embed, rate=self.dropout_rate)

                tf.get_variable_scope().reuse_variables()
                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1_dropout(padding, state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2_dropout(tf.concat([output1, current_embed], 1), state2)

                labels = tf.convert_to_tensor(caption[:, i])  #### batch_size x 1 ####### i correspond to current word
                onehot_labels = tf.one_hot(labels, self.n_words, 1.0, 0.0) ### batch_size * word_number

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                softmax_value = tf.nn.log_softmax(logit_words,dim=-1)#### batch_size x word_number
                loss.append(tf.multiply(softmax_value*onehot_labels, tf.expand_dims(caption_mask[:,i],1))) ####t x batch_size x word_number
                ##different from loss.append( tf.transpose(tf.mul(tf.transpose(tf.log(tf.clip_by_value(softmax, 1e-20, 1.0)) * tf.one_hot(captions[:, t], 23111), [1, 0]),  mask[:, t]), [1, 0])
            #loss = self.loss_weight * tf.transpose(tf.stack(loss),[1,0,2]) ### batch_size x t x word_number
            loss = tf.transpose(tf.stack(loss), [1, 0, 2])
	######## multilabel loss########
        attribute_feature = tf.reduce_mean(video, axis=1)
        logits = tf.nn.xw_plus_b(attribute_feature, self.attr_W, self.attr_b)
        sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_labels, logits=logits)
	multilabel_loss = tf.reduce_sum(sigmoid_cross_entropy)/float(self.label_dim * self.batch_size)
        return loss, video_frames, caption, caption_mask, true_labels, multilabel_loss

    def build_multinomial_sampler(self):
        ########## inception resnet v2####
        ###preprocessing###
        video_frames = tf.placeholder(tf.float32,
                                      [None, self.n_video_lstm_step, self.height, self.width, self.channels])
        all_frames = tf.reshape(video_frames, [-1, self.height, self.width, self.channels])
        # true_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim])

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2',
                                   reuse=None) as scope:
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=False):
                    tf.get_variable_scope().reuse_variables()
                    net, endpoints = inception_resnet_v2_base(all_frames, scope=scope)
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
                    net = slim.dropout(net, self.dropout_rate, is_training=True, scope='Dropout')
        video = tf.reshape(net, [-1, self.n_video_lstm_step, self.dim_image])

        #####################################################

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_lstm_step, self.word_dim])
        state1 = tf.zeros([self.batch_size, self.lstm1.state_size], tf.float32)
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size], tf.float32)
        padding = tf.zeros([self.batch_size, self.word_dim], tf.float32)

        sampled_words = []
        seqlogprobs = []
        with tf.variable_scope("s2vt") as scope:
            for i in range(0, self.n_video_lstm_step):
                #if i > 0:
                tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(image_emb[:, i, :], state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([output1, padding], 1), state2)

                    ############### decoding ##########
                    # with tf.variable_scope("s2vt") as scope:
            for i in range(0, self.n_caption_lstm_step):
                tf.get_variable_scope().reuse_variables()
                if i ==0:
                    with tf.device('/cpu:0'):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([self.batch_size],dtype=tf.int64))
                else:
                    with tf.device('/cpu:0'):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, sampled_word)
                        #current_embed = tf.expand_dims(current_embed, 0)

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding, state1)
                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([output1, current_embed], 1), state2)
                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                word_probability = tf.nn.log_softmax(logit_words)### batch_size x word_dim

                sampled_word = tf.multinomial(word_probability, 1) ### batchsize x 1
                sampled_word = tf.reshape(sampled_word, [-1])
                sampled_words.append(sampled_word)

            sampled_captions = tf.transpose(tf.stack(sampled_words),[1,0])

        return sampled_captions,video_frames

        # return video, tf_sentence
    def build_sampler(self):
        video_frames = tf.placeholder(tf.float32,
                                      [None, self.n_video_lstm_step, self.height, self.width, self.channels])
        all_frames = tf.reshape(video_frames, [-1, self.height, self.width, self.channels])

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2',
                                   reuse=None) as scope:
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=False):
                    tf.get_variable_scope().reuse_variables() ### uncomment in training
                    net, endpoints = inception_resnet_v2_base(all_frames, scope=scope)
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
                    net = slim.dropout(net, self.dropout_rate, is_training=False, scope='Dropout')
        video = tf.reshape(net, [-1, self.n_video_lstm_step, self.dim_image])
        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        #image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_lstm_step, self.word_dim])
        #state1 = tf.zeros([self.batch_size, self.lstm1.state_size], tf.float32)
        #state2 = tf.zeros([self.batch_size, self.lstm2.state_size], tf.float32)
        #padding = tf.zeros([self.batch_size, self.word_dim], tf.float32)
        image_emb = tf.reshape(image_emb, [-1, self.n_video_lstm_step, self.word_dim])

        state1 = tf.zeros(tf.stack([tf.shape(video)[0], self.lstm1.state_size]), tf.float32)
        state2 = tf.zeros(tf.stack([tf.shape(video)[0], self.lstm2.state_size]), tf.float32)
        padding = tf.zeros(tf.stack([tf.shape(video)[0], self.word_dim]), tf.float32)

        sampled_words = []
        probs = []
        with tf.variable_scope("s2vt") as scope:
            for i in range(0, self.n_video_lstm_step):
                #if i > 0: uncomment in training
                tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(image_emb[:, i, :], state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([output1, padding], 1), state2)

                    ############### decoding ##########
                    # with tf.variable_scope("s2vt") as scope:
            for i in range(0, self.n_caption_lstm_step):
                tf.get_variable_scope().reuse_variables()
                if i ==0:
                    with tf.device('/cpu:0'):
                        #current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([self.batch_size],dtype=tf.int64))
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([tf.shape(video)[0]], dtype=tf.int64))
                else:
                    with tf.device('/cpu:0'):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, sampled_word)
                        #current_embed = tf.expand_dims(current_embed, 0)

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding, state1)
                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([output1, current_embed], 1), state2)
                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                sampled_word = tf.argmax(logit_words,1)
                sampled_words.append(sampled_word)

            sampled_captions = tf.transpose(tf.stack(sampled_words),[1,0])
        return sampled_captions,video_frames

    def build_mix_sample(self):
        video_frames = tf.placeholder(tf.float32,
                                      [None, self.n_video_lstm_step, self.height, self.width, self.channels])
        all_frames = tf.reshape(video_frames, [-1, self.height, self.width, self.channels])

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2',
                                   reuse=None) as scope:
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=False):
                    tf.get_variable_scope().reuse_variables()
                    net, endpoints = inception_resnet_v2_base(all_frames, scope=scope)
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
                    net = slim.dropout(net, self.dropout_rate, is_training=False, scope='Dropout')
        video = tf.reshape(net, [-1, self.n_video_lstm_step, self.dim_image])
        video_flat = tf.reshape(video, [-1, self.dim_image])
        caption = tf.placeholder(tf.int32, [self.batch_size,
                                            self.n_caption_lstm_step])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        # image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_lstm_step, self.word_dim])
        # state1 = tf.zeros([self.batch_size, self.lstm1.state_size], tf.float32)
        # state2 = tf.zeros([self.batch_size, self.lstm2.state_size], tf.float32)
        # padding = tf.zeros([self.batch_size, self.word_dim], tf.float32)
        image_emb = tf.reshape(image_emb, [-1, self.n_video_lstm_step, self.word_dim])

        state1 = tf.zeros(tf.stack([tf.shape(video)[0], self.lstm1.state_size]), tf.float32)
        state2 = tf.zeros(tf.stack([tf.shape(video)[0], self.lstm2.state_size]), tf.float32)
        padding = tf.zeros(tf.stack([tf.shape(video)[0], self.word_dim]), tf.float32)

        sampled_words = []
        probs = []

        ##################### compute choose word probability###
        # k = tf.convert_to_tensor([self.k_value],dtype=tf.float64)
        onehundred_percent = tf.convert_to_tensor([1.00001], dtype=tf.float64)

        # step_num = tf.cast(steps,dtype=tf.float64)
        # print 'inner steps: ',self.steps
        # true_word_prob = tf.expand_dims(tf.divide(k,tf.add(k,tf.exp(tf.divide(step_num,k)))),0)

        true_word_prob = tf.expand_dims(tf.convert_to_tensor([0.9], dtype=tf.float64), 0)
        pre_prob = tf.concat([true_word_prob, tf.subtract(onehundred_percent, true_word_prob)], 1)
        probabilities = tf.tile(pre_prob, [self.batch_size, 1])
        log_probs = tf.log(probabilities)
        row_indice = tf.cast(tf.expand_dims(tf.range(0, self.batch_size), 1), tf.int64)

        # previous_words = tf.zeros(self.batch_size)

        with tf.variable_scope("s2vt") as scope:
            for i in range(0, self.n_video_lstm_step):
                # if i > 0:
                tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(image_emb[:, i, :], state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([output1, padding], 1), state2)

                    ############### decoding ##########
                    # with tf.variable_scope("s2vt") as scope:
            for i in range(0, self.n_caption_lstm_step):
                tf.get_variable_scope().reuse_variables()
                indice0 = tf.multinomial(log_probs, num_samples=1)
                indice = tf.concat([row_indice, indice0], 1)
                if i == 0:
                    with tf.device('/cpu:0'):
                        # current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([self.batch_size],dtype=tf.int64))
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([tf.shape(video)[0]], dtype=tf.int64))
                else:
                    sampled_word = tf.stop_gradient(sampled_word)
                    words = tf.concat([tf.expand_dims(caption[:, i - 1], 1), tf.expand_dims(sampled_word, 1)], 1)
                    previous_words = tf.gather_nd(words, indice)
                    with tf.device('/cpu:0'):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, previous_words)
                        # current_embed = tf.expand_dims(current_embed, 0)

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding, state1)
                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([output1, current_embed], 1), state2)
                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                sampled_word = tf.cast(tf.argmax(logit_words, 1),tf.int32)
                sampled_words.append(sampled_word)

            sampled_captions = tf.transpose(tf.stack(sampled_words), [1, 0])
        return sampled_captions, video_frames, caption

    def evaluate_multilabel(self, threshold):

        video_frames = tf.placeholder(tf.float32,
                                  [None, self.n_video_lstm_step, self.height, self.width, self.channels])
        all_frames = tf.reshape(video_frames, [-1, self.height, self.width, self.channels])

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2',reuse = None) as scope:
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                    is_training = False):
                    tf.get_variable_scope().reuse_variables()
                    net, endpoints = inception_resnet_v2_base(all_frames, scope=scope)
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
                    net = slim.dropout(net, self.dropout_rate, is_training=False, scope='Dropout')
        video = tf.reshape(net, [-1, self.n_video_lstm_step, self.feature_dim])
        attribute_feature = tf.reduce_mean(video, axis=1)
        logits = tf.nn.xw_plus_b(attribute_feature, self.attr_W, self.attr_b)
        scores = tf.sigmoid(logits)#### batch_size x label_num

        return video_frames, scores
# =====================================================================================
# Global Parameters
# =====================================================================================

# video_train_caption_file = './data/video_corpus.csv'
# video_test_caption_file = './data/video_corpus.csv'

#model_path = './vocab1_models'
model_path = './rouge_reinforcement_multitask_models'

video_path = '/data1/lijun/msvd/video-frames'

#video_train_feature_file = '/media/llj/storage/all_sentences/msvd_inception_globalpool_train_origin.txt'

#video_test_feature_file = '/media/llj/storage/all_sentences/msvd_inception_globalpool_test_origin.txt'

video_train_sent_file = '/data1/lijun/tensorflow_s2vt/msvd_sents_train_noval_lc_nopunc.txt'

video_test_sent_file = '/data1/lijun/tensorflow_s2vt/msvd_sents_test_lc_nopunc.txt'

#vocabulary_file = '/media/llj/storage/all_sentences/coco_msvd_allvocab.txt'
vocabulary_file = '/data1/lijun/tensorflow_s2vt/msvd_vocabulary1.txt'

vocab_file = '/data1/lijun/tensorflow_s2vt/train_most_freq_vocab_400_truncated.txt' ##### attribute labels

feature_model = '/data1/lijun/tensorflow_s2vt/inception_resnet_v2_2016_08_30.ckpt'
lstm_model ='/data1/lijun/tensorflow_s2vt/new_rouge_multisamp_reinforcement_models/rouge_reinforce_multisample8_model-6'

#reinforcement_model = '/home/llj/tensorflow_s2vt/reinforcement_multitask_models/10batch_size2reinforce_multitask_model_lambda-198000'

multitask_model ='/home/lijun/tensor_examples/reinforcement_multitask_models/5batch_size8reinforce_multitask_model_alpha01-7500'

#out_file = 'multitask_models/2batch_scores_10_noval_alpha0.01_e2e_continue.txt'
out_file = 'rouge_reinforcement_multitask_models/reinf_multi_alpha005_test.txt'

save_model_name = 'reinforce_multitask_model_alpha005'

save_loss_imgs = 'reinforce_multitask_alpha005'

save_cider_imgs = 'reinforce_multitask_alpha005'

highest_cider = 0

# =======================================================================================
# Train Parameters
# =======================================================================================
dim_image = 1536
lstm_dim = 1000
word_dim = 500

n_lstm_step = 40
n_caption_lstm_step = 35
n_video_lstm_step = 5

n_epochs = 20
batch_size = 16
#start_learning_rate = 0.0001
start_learning_rate = 0.000001
width = 299
height = 299
channels = 3

feature_dim = dim_image
nums_label = 400
threshold = 0.5
num_videos = n_video_lstm_step
end_iter = 500
iteration_size = 1000

lambda_loss = 0.5
alpha = 0.05
#caption_mask_out = open('caption_masks.txt', 'w')

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

def get_video_feature_caption_pair(sent_file=video_train_sent_file, frame_path=video_path, num_frame_per_video = n_video_lstm_step,prefix=''):
    sents = []
    vid = []
    video_frames = {}
    with open(sent_file, 'r') as video_sent_file:
        for line in video_sent_file:
            line = line.strip()
            id_sent = line.split('\t')
            sents.append((id_sent[0], id_sent[1]))
            if id_sent[0] not in vid:
                vid.append(id_sent[0])
    for vid_name in vid:
        video_frames[vid_name] = []
        video_path = frame_path + '/' + vid_name
        frame_cnt = len(glob.glob(video_path+'/'+prefix+'*'))
        step = (frame_cnt-2)//(num_frame_per_video-1)
        if step >0 :
            frame_ticks = range(1, min((2 + step * (num_frame_per_video-1)), frame_cnt), step)
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
    sents = np.array(sents)
    return sents, video_frames, vid

def later_get_video_feature_caption_pair(sent_file=video_train_sent_file, frame_path=video_path, num_frame_per_video = n_video_lstm_step,prefix='frame_'):
    sents = []
    vid = []
    video_frames = {}
    with open(sent_file, 'r') as video_sent_file:
        for line in video_sent_file:
            line = line.strip()
            id_sent = line.split('\t')
            sents.append((id_sent[0], id_sent[1]))
            if id_sent[0] not in vid:
                vid.append(id_sent[0])
    for vid_name in vid:
        video_frames[vid_name] = []
        video_path = frame_path + '/' + vid_name
        frame_cnt = len(glob.glob(video_path+'/'+prefix+'*'))
        step = (frame_cnt-1)//(num_frame_per_video)
        if step >0 :
            frame_ticks = range(frame_cnt, max((frame_cnt - step * (num_frame_per_video-1)-1), 1), -step)
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
    sents = np.array(sents)
    return sents, video_frames, vid


def preProBuildWordVocab(vocabulary, word_count_threshold=0):
    # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold)
    word_counts = {}
    nsents = 0
    vocab = vocabulary

    ixtoword = {}
    # ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[0] = '<eos>'

    wordtoix = {}
    # wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 0

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx + 2
        ixtoword[idx + 2] = w

    return wordtoix, ixtoword

def image_reading_processing(path):
    video_batch = [[] for x in xrange(len(path))]
    for i in xrange(len(path)):
        for j in xrange(n_video_lstm_step):
            image = cv2.imread(path[i][j], cv2.IMREAD_COLOR)### height,width,channels
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
            image = image.astype(np.float32)
            image = 2 * (image/255.0) - 1

            video_batch[i].append(image)
    return video_batch

def sentence_padding_toix(captions_batch, wordtoix):  ###########return dimension is n_caption_lstm_step
    captions_mask = []
    for idx, each_cap in enumerate(captions_batch):
        one_caption_mask = np.ones(n_caption_lstm_step)
        word = each_cap.lower().split(' ')
        if len(word) < n_caption_lstm_step:
            for i in range(len(word), n_caption_lstm_step):
                captions_batch[idx] = captions_batch[idx] + ' <eos>'
                if i != len(word):
                    one_caption_mask[i] = 0
        else:
            new_word = ''
            for i in range(n_caption_lstm_step - 1):
                new_word = new_word + word[i] + ' '
            captions_batch[idx] = new_word + '<eos>'
        # one_caption_mask=np.reshape(one_caption_mask,(-1,n_caption_lstm_step))
        captions_mask.append(one_caption_mask)
    captions_mask = np.reshape(captions_mask, (-1, n_caption_lstm_step))
    caption_batch_ind = []
    for cap in captions_batch:
        current_word_ind = []
        for word in cap.lower().split(' '):
            if word in wordtoix:
                current_word_ind.append(wordtoix[word])
            else:
                current_word_ind.append(wordtoix['<en_unk>'])
        # current_word_ind.append(0)###make one more dimension
        caption_batch_ind.append(current_word_ind)
    i = 0
    #caption_mask_out.write('captions: ' + str(caption_batch_ind) + '\n' + 'masks: ' + str(captions_mask) + '\n')
    return caption_batch_ind, captions_mask

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

def get_captions(captions,vid):
    return [y for x,y in captions if x == vid]

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

def train():  ###### move caption (input_sentence) one column left and also need to move caption_mask (cont_sent)one column left ########################################################llj
    ######### read caption, video frame path, video id######
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    train_captions, video_frames, train_vid = get_video_feature_caption_pair(video_train_sent_file, video_path, num_frame_per_video=n_video_lstm_step)
    vocabulary = []
    test_captions, test_video_frames, test_vid = get_video_feature_caption_pair(video_test_sent_file, video_path, num_frame_per_video=n_video_lstm_step)
    ##### read attribute
    train_vid_sent, vocab, label_num = read_sent_vocab_file(video_train_sent_file, vocab_file)
    test_vid_sent, _, _ = read_sent_vocab_file(video_test_sent_file, vocab_file)
    assert nums_label == label_num, 'vocab file label number is not equal to nums_label'

    train_labels = get_multilabel(train_vid_sent, vocab)
    test_labels = get_multilabel(test_vid_sent, vocab)
    #######################

    with open(vocabulary_file, 'r') as vocab:
        for line in vocab:
            vocabulary.append(line.rstrip())

    wordtoix, ixtoword = preProBuildWordVocab(vocabulary, word_count_threshold=0)

    if not os.path.exists('./new_vocab1_data/wordtoix') or os.path.exists('./new_vocab1_data/ixtoword'):
        np.save("./new_vocab1_data/wordtoix", wordtoix)
        np.save('./new_vocab1_data/ixtoword', ixtoword)

    model = Video_Caption_Generator(
        dim_image=dim_image,
        n_words=len(wordtoix),
        word_dim=word_dim,
        lstm_dim=lstm_dim,
        batch_size=batch_size,
        n_lstm_steps=n_lstm_step,
        n_video_lstm_step=n_video_lstm_step,
        n_caption_lstm_step=n_caption_lstm_step,
        bias_init_vector=None,
	alpha = alpha)

    model_loss, model_features, model_captions, model_caption_masks, _,model_multilabel_loss = model.build_model()
    tf_test_video_frames, tf_scores = model.evaluate_multilabel(threshold=threshold)
    sampled_captions, multinomial_video_features = model.build_multinomial_sampler() ################### multinomial
    #sampled_captions, multinomial_video_features, mix_captions = model.build_mix_sample() ###### 0.9 ground truth baseline
    greedy_captions, greedy_video_features = model.build_sampler()
    rewards = tf.placeholder(tf.float32, [None])
    base_line = tf.placeholder(tf.float32, [None])
    #sampled_caption_mask = tf.placeholder(tf.int32, [None, n_caption_lstm_step]) ### batch_size x n_caption_lstm_step
    loss, loss_features, loss_captions, loss_masks,true_labels, multilabel_loss = model.build_loss()


    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.InteractiveSession(config=config)

    # my tensorflow version is 0.12.1, I write the saver with version 1.0
    saver = tf.train.Saver(max_to_keep=100, write_version=1)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                               15000, 0.5, staircase=True)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    norm = tf.reduce_sum(loss_masks)
    residual = rewards - base_line
    #residual = tf.clip_by_value(residual,-1,1) ######## reward clipping
    sum_loss = -(1-alpha)*tf.reduce_sum(tf.transpose(tf.multiply(tf.transpose(loss,[2,1,0]), residual),[2,1,0]))/norm + alpha * multilabel_loss ############# loss function
#lambda_loss * model_loss### transpose loss:batch x video_step_word_dim
    #### every gradient clipping
    #gvs = optimizer.compute_gradients(sum_loss)
    #capped_gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
    #train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    ### global clipping
    grads_rl, _ = tf.clip_by_global_norm(tf.gradients(sum_loss, tf.trainable_variables(), aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N), 10)
    grads_and_vars = list(zip(grads_rl,tf.trainable_variables()))
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars,global_step=global_step)

    ############## clipping every gradient####
    #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss, global_step=global_step)
    tf.global_variables_initializer().run()
    #tf.summary.scalar('lr',learning_rate)
    #optimistic_restore(sess, feature_model)
    #optimistic_restore(sess, lstm_model)
    optimistic_restore(sess,feature_model)
    optimistic_restore(sess,lstm_model)
    #optimistic_restore(sess,multitask_model)
    #optimistic_restore(sess, reinforcement_model)

    #optimistic_restore(sess, '/home/llj/tensorflow_s2vt/e2e_models/10batch_size2cnn_model_false_withval-15000')
    # new_saver = tf.train.Saver()
    # new_saver = tf.train.import_meta_graph('./rgb_models/model-1000.meta')
    # new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))

    loss_to_draw = []
    cider_to_draw = []
   # summary_op = tf.summary.merge_all()
    ############# test before training########

    #greedy_captions, greedy_video_features = model.build_sampler()
    ######

    ######################################### caption evaluation ##########################
    with open(out_file, 'a') as f:
        all_decoded_for_eval = {}
        test_index = list(range(len(test_captions)))
        random.shuffle(test_index)
        ref_decoded = {}
        for aa in xrange(0,len(set(test_captions[:,0])),batch_size):

            id = list(set(test_captions[:,0]))[aa:aa+batch_size]
            test_video_frames_batch = [test_video_frames[x] for x in id]
            test_video_batch = image_reading_processing(test_video_frames_batch)

            feed_dict = {greedy_video_features: test_video_batch}
            greedy_words = sess.run(greedy_captions, feed_dict) #### batch_size x num of each words
            greedy_decoded = decode_captions(np.array(greedy_words), ixtoword)
            for videoid in id:
                if videoid not in all_decoded_for_eval:
                    all_decoded_for_eval[videoid] = []

            [all_decoded_for_eval[x].append(y) for x,y in zip(id,greedy_decoded)]

        for num in xrange(0, len(test_captions),batch_size):

            videoid = test_captions[num:num+batch_size,0]
            for id in videoid:
                if id not in ref_decoded:
                    ref_decoded[id] = []
            [ref_decoded[x].append(y) for x,y in zip(videoid,test_captions[num:num+batch_size,1])]

        scores = evaluate_for_particular_captions(all_decoded_for_eval, ref_decoded)

        f.write('before train: ')
        f.write('\n')
        f.write("Bleu_1:" + str(scores['Bleu_1']))
        f.write('\n')
        f.write("Bleu_2:" + str(scores['Bleu_2']))
        f.write('\n')
        f.write("Bleu_3:" + str(scores['Bleu_3']))
        f.write('\n')
        f.write("Bleu_4:" + str(scores['Bleu_4']))
        f.write('\n')
        f.write("ROUGE_L:" + str(scores['ROUGE_L']))
        f.write('\n')
        f.write("CIDEr:" + str(scores['CIDEr']))
        f.write('\n')
        #f.write("METEOR:" + str(scores['METEOR']))
        #f.write('\n')
        #f.write("metric:" + str(
        #    1 * scores['METEOR'] ))
        f.write('\n')
	######## multilabel ######
        scores = []
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        count_pos = 0
        count_neg = 0
        for aa in xrange(0, len(test_vid), batch_size):
            id = test_vid[aa:aa + batch_size]
            num_videos = len(id)
            test_sample_labels = [test_labels[x] for x in id]
            test_samples = [test_video_frames[x] for x in id]
            test_video_batch = image_reading_processing(test_samples)

            scores = sess.run(tf_scores, feed_dict={tf_test_video_frames: test_video_batch})
            temp_true_positive, temp_true_negative, temp_false_positive, temp_false_negative, temp_count_pos, temp_count_neg = get_metrics(
                scores,
                test_sample_labels, num_videos)
            true_positive += temp_true_positive
            true_negative += temp_true_negative
            false_positive += temp_false_positive
            false_negative += temp_false_negative
            count_pos += temp_count_pos
            count_neg += temp_count_neg
        sensitivity = true_positive / float(count_pos) if count_pos > 0 else 0
        specificity = true_negative / float(count_neg) if count_neg > 0 else 0
        harmmean = 2.0 / (count_pos / float(true_positive) + count_neg / float(true_negative)) if (
            (count_pos + count_neg) > 0) else 0
        precision = (true_positive / float(true_positive + false_positive)) if (true_positive > 0) else 0
        f1_score = 2.0 * true_positive / float(2 * true_positive + false_positive + false_negative) if (
            true_positive > 0) else 0
        f.write('before train: ')
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
######################################### caption evaluation ##########################

    iter_count = 0
    early_stopping = 0 #### stop sign
    highest_cider = 0 ####

    for epoch in range(0, n_epochs):
        loss_to_draw_epoch = []

        ## randomize the video id order
	random.seed(1)
        index = list(range(len(train_captions)))
        random.shuffle(index)
        ### iterate over the video id
        for start, end in zip(range(0, len(index)-batch_size, batch_size), range(batch_size, len(index), batch_size)):
            iter_count += 1
            ref_decoded = defaultdict()
            start_time = time.time()
            vid, sentence = train_captions[index[start:end], 0], train_captions[index[start:end], 1]
            captions_batch = sentence.tolist()
            video_frames_batch = [video_frames[x] for x in vid]
            video_batch = image_reading_processing(video_frames_batch)
            # captions_batch = map(lambda x: '<bos> ' + x, captions_batch)
            captions_ind, captions_mask = sentence_padding_toix(captions_batch, wordtoix)
	############### multilabel ############
            sample_labels = [train_labels[x] for x in vid]

            samples, greedy_words = sess.run([sampled_captions,greedy_captions],feed_dict={
                                                            multinomial_video_features: video_batch,
                                                            greedy_video_features: video_batch})
            mask, multi_decoded = decode_captions_masks(samples,ixtoword)
            greedy_mask, greedy_decoded = decode_captions_masks(greedy_words,ixtoword)
            ###ground truth captions
            for i, video_id in enumerate(vid):
                ref_decoded[i] = get_captions(train_captions, video_id)
            #ref_decoded = [decode_captions(np.array(captions_ind[j]),ixtoword) for j in range(len(captions_ind))]
            r = evaluate_captions_cider(ref_decoded,multi_decoded)
            b = evaluate_captions_cider(ref_decoded,greedy_decoded)
            # ref_decoded = [train_captions[train_captions[:,0] == x, 1].tolist() for x in vid]
           # r = [evaluate_captions([k],[v]) for k,v in zip(ref_decoded, multi_decoded)]
            #r =pool.map(evaluate_captions_wrapper, zip(ref_decoded, multi_decoded))
            #b = [evaluate_captions([k],[v]) for k,v in zip(ref_decoded, greedy_decoded)]
            #b = pool.map(evaluate_captions_wrapper, zip(ref_decoded, greedy_decoded))
            #loss_to_draw_epoch.extend(b)
            print 'r-b: ', map(operator.sub, r, b)
            print 'r-b: ',np.mean(np.array(r) - np.array(b))

            feed_dict = {loss_masks: mask, loss_captions: samples, loss_features: video_batch, rewards: r, base_line: b, model_features: video_batch, model_captions: captions_ind, model_caption_masks: captions_mask, true_labels: sample_labels}
            _,loss_val = sess.run([train_op,sum_loss],feed_dict)
            #loss_to_draw_epoch.extend(b)
            loss_to_draw_epoch.append(loss_val)


            if np.mod(iter_count, iteration_size) == 0:
                with open(out_file, 'a') as f:
                    all_decoded_for_eval = {}
                    test_index = list(range(len(test_captions)))
                    random.shuffle(test_index)
                    ref_decoded = {}
                    for aa in xrange(0, len(set(test_captions[:, 0])), batch_size):

                        id = list(set(test_captions[:, 0]))[aa:aa + batch_size]
                        test_video_frames_batch = [test_video_frames[x] for x in id]
                        test_video_batch = image_reading_processing(test_video_frames_batch)

                        feed_dict = {greedy_video_features: test_video_batch}
                        greedy_words = sess.run(greedy_captions, feed_dict)  #### batch_size x num of each words
                        greedy_decoded = decode_captions(np.array(greedy_words), ixtoword)
                        for videoid in id:
                            if videoid not in all_decoded_for_eval:
                                all_decoded_for_eval[videoid] = []

                        [all_decoded_for_eval[x].append(y) for x, y in zip(id, greedy_decoded)]

                    for num in xrange(0, len(test_captions), batch_size):

                        videoid = test_captions[num:num + batch_size, 0]
                        for id in videoid:
                            if id not in ref_decoded:
                                ref_decoded[id] = []
                        [ref_decoded[x].append(y) for x, y in zip(videoid, test_captions[num:num + batch_size, 1])]

                    scores = evaluate_for_particular_captions(all_decoded_for_eval, ref_decoded)

                    f.write('Epoch %d\n' % epoch)
                    f.write("b:" + str(np.mean(np.array(loss_to_draw_epoch))))
                    f.write('\n')
                    f.write('iteration %d\n' % iter_count)
                    f.write('\n')
                    f.write("Bleu_1:" + str(scores['Bleu_1']))
                    f.write('\n')
                    f.write("Bleu_2:" + str(scores['Bleu_2']))
                    f.write('\n')
                    f.write("Bleu_3:" + str(scores['Bleu_3']))
                    f.write('\n')
                    f.write("Bleu_4:" + str(scores['Bleu_4']))
                    f.write('\n')
                    f.write("ROUGE_L:" + str(scores['ROUGE_L']))
                    f.write('\n')
                    f.write("CIDEr:" + str(scores['CIDEr']))
                    f.write('\n')
                    f.write("METEOR:" + str(scores['METEOR']))
                    f.write('\n')
                    f.write("metric:" + str(
                        1 * scores['METEOR']))
                    f.write('\n')
                    f.write('\n')
                print 'CIDEr: ',scores['CIDEr']
                if highest_cider < scores['CIDEr']:
                    highest_cider = scores['CIDEr']
                    early_stopping = 0
                cider_to_draw.append(scores['CIDEr'])

                with open(out_file, 'a') as f:
                    scores = []
                    true_positive = 0
                    true_negative = 0
                    false_positive = 0
                    false_negative = 0
                    count_pos = 0
                    count_neg = 0
                    for aa in xrange(0, len(test_vid), batch_size):
                        id = test_vid[aa:aa + batch_size]
                        num_videos = len(id)
                        test_sample_labels = [test_labels[x] for x in id]
                        test_samples = [test_video_frames[x] for x in id]
                        test_video_batch = image_reading_processing(test_samples)

                        scores = sess.run(tf_scores, feed_dict={tf_test_video_frames: test_video_batch})
                        temp_true_positive, temp_true_negative, temp_false_positive, temp_false_negative, temp_count_pos, temp_count_neg = get_metrics(
                            scores,
                            test_sample_labels, num_videos)
                        true_positive += temp_true_positive
                        true_negative += temp_true_negative
                        false_positive += temp_false_positive
                        false_negative += temp_false_negative
                        count_pos += temp_count_pos
                        count_neg += temp_count_neg
                    sensitivity = true_positive / float(count_pos) if count_pos > 0 else 0
                    specificity = true_negative / float(count_neg) if count_neg > 0 else 0
                    harmmean = 2.0 / (count_pos / float(true_positive) + count_neg / float(true_negative)) if (
                    (count_pos + count_neg) > 0) else 0
                    precision = (true_positive / float(true_positive + false_positive)) if (true_positive > 0) else 0
                    f1_score = 2.0 * true_positive / float(2 * true_positive + false_positive + false_negative) if (
                    true_positive > 0) else 0
                    f.write('Epoch %d\n' % epoch)
                    f.write('\n')
                    f.write('alpha: ' + str(alpha))
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

            if np.mod(iter_count, iteration_size) == 0:
                print "iter_count ", iter_count, " is done. Saving the model ..."
                saver.save(sess, os.path.join(model_path,
                                              str(n_video_lstm_step) + 'batch_size' + str(batch_size) + save_model_name),
                           global_step=iter_count)
                if early_stopping > end_iter:
                    return
                early_stopping += 1
                print 'highest cider: ',highest_cider

            print 'idx: ', start, ' rate: ', sess.run(learning_rate)," Epoch: ", epoch, " loss: ", np.mean(np.array(loss_val)),\
                ' Elapsed time: ', str((time.time() - start_time))





        # draw loss curve every epoch


def test(model_path='/home/llj/tensorflow_s2vt/vocab1_models/'):
    test_captions, video_frames, _ = get_video_feature_caption_pair(video_test_sent_file, video_path, num_frame_per_video=n_video_lstm_step)

    ixtoword = pd.Series(np.load('./vocab1_data/ixtoword.npy').tolist())

    model = Video_Caption_Generator(
        dim_image=dim_image,
        n_words=len(ixtoword),
        word_dim=word_dim,
        lstm_dim=lstm_dim,
        batch_size=batch_size,
        n_lstm_steps=n_lstm_step,
        n_video_lstm_step=n_video_lstm_step,
        n_caption_lstm_step=n_caption_lstm_step,
        bias_init_vector=None)

    # video_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    video_tf, captions_tf, logprob_tf = model.build_generator()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.InteractiveSession(config=config)

    for i in xrange(20):
        model_path_last = model_path + str(n_video_lstm_step)+'batch_size'+ str(batch_size)+'cnn_model' + str(i)
        out_file = 'vocab9960_batchsize_'+str(batch_size)+'_ep'+ str(i) + '.txt'
        saver = tf.train.Saver()
        saver.restore(sess, model_path_last)

        test_output_txt_fd = open(out_file, 'w')
        for key, values in video_frames.iteritems():
            video_frames_batch = [video_frames[key]]
            video_batch = image_reading_processing(video_frames_batch)
            generated_word_index = sess.run(captions_tf, feed_dict={video_tf: video_batch})
            generated_words = ixtoword[generated_word_index]

            punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
            generated_words = generated_words[:punctuation]

            generated_sentence = ' '.join(generated_words)
            generated_sentence = generated_sentence.replace('<bos> ', '')
            generated_sentence = generated_sentence.replace(' <eos>', '')
            print generated_sentence, '\n'
            test_output_txt_fd.write(key + '\t')
            test_output_txt_fd.write(generated_sentence + '\n')

def evaluation(model_path='/data1/lijun/tensorflow_s2vt/reinforcement_multitask_models/basedon_reinforce8multisampling_sigmoidloss_reinforceloss/'):#home/llj/tensorflow_s2vt/reinforcement_multitask_models/'):
    start_time = time.time()
    test_captions, test_video_frames, _ = get_video_feature_caption_pair(video_test_sent_file, video_path,
                                                                      num_frame_per_video=n_video_lstm_step)

    ixtoword = pd.Series(np.load('./vocab1_data/ixtoword.npy').tolist())
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    #model_path_last = model_path + '10batch_size2reinforce_multitask_model_lambda0.2-9000'


    model = Video_Caption_Generator(
        dim_image=dim_image,
        n_words=len(ixtoword),
        word_dim=word_dim,
        lstm_dim=lstm_dim,
        batch_size=batch_size,
        n_lstm_steps=n_lstm_step,
        n_video_lstm_step=n_video_lstm_step,
        n_caption_lstm_step=n_caption_lstm_step,
        bias_init_vector=None)
    greedy_captions, greedy_video_features = model.build_sampler()

    saver = tf.train.Saver()

    #saver.restore(sess, model_path_last)

    with open('reinforcement_multitask_models/reinforce_multitask_basedon_8reinfo_multisamp_test1.txt', 'a') as f:
      for i in xrange(44000, 45000, 1000):
        model_path_last = model_path + '5batch_size8reinforce_multitask_model_alpha005-' + str(i)
        saver.restore(sess, model_path_last)
        all_decoded_for_eval = {}
        test_index = list(range(len(test_captions)))
        random.shuffle(test_index)
        ref_decoded = {}
        process = psutil.Process(os.getpid())
        print('process memory : ', process.memory_info().rss)
        for aa in xrange(0,len(set(test_captions[:,0])),batch_size):

            id = list(set(test_captions[:,0]))[aa:aa+batch_size]
            test_video_frames_batch = [test_video_frames[x] for x in id]
            test_video_batch = image_reading_processing(test_video_frames_batch)

            feed_dict = {greedy_video_features: test_video_batch}
            greedy_words = sess.run(greedy_captions, feed_dict) #### batch_size x num of each words
            greedy_decoded = decode_captions(np.array(greedy_words), ixtoword)
            for videoid in id:
                if videoid not in all_decoded_for_eval:
                    all_decoded_for_eval[videoid] = []

            [all_decoded_for_eval[x].append(y) for x,y in zip(id,greedy_decoded)]
        end_time = time.time() -start_time
        print('testing time : ',end_time)

        for num in xrange(0, len(test_captions),batch_size):

            videoid = test_captions[num:num+batch_size,0]
            for id in videoid:
                if id not in ref_decoded:
                    ref_decoded[id] = []
            [ref_decoded[x].append(y) for x,y in zip(videoid,test_captions[num:num+batch_size,1])]

        scores = evaluate_for_particular_captions(all_decoded_for_eval, ref_decoded)

        f.write('iteration: ' + str(i))
        f.write('\n')
        f.write("Bleu_1:" + str(scores['Bleu_1']))
        f.write('\n')
        f.write("Bleu_2:" + str(scores['Bleu_2']))
        f.write('\n')
        f.write("Bleu_3:" + str(scores['Bleu_3']))
        f.write('\n')
        f.write("Bleu_4:" + str(scores['Bleu_4']))
        f.write('\n')
        f.write("ROUGE_L:" + str(scores['ROUGE_L']))
        f.write('\n')
        f.write("CIDEr:" + str(scores['CIDEr']))
        f.write('\n')
        f.write("METEOR:" + str(scores['METEOR']))
        f.write('\n')
        f.write("metric:" + str(
            1 * scores['METEOR'] ))
        f.write('\n')
        f.write('\n')
        print 'number model: ', i
    print 'CIDEr: ', scores['CIDEr']

if __name__ == '__main__':
    args = parse_args()
    if args.task == 'train':
        with tf.device('/gpu:' + str(args.gpu_id)):
            train()
    elif args.task == 'test':
        with tf.device('/gpu:' + str(args.gpu_id)):
            test()
    elif args.task == 'evaluate':
        with tf.device('/gpu:' + str(args.gpu_id)):
            evaluation()
