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
from cider_evaluation import *
#from evaluation import *
import operator
from multiprocessing import Pool



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Extract a CNN features')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
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
                 n_caption_lstm_step, bias_init_vector=None, loss_weight = 1, decay_value = 0.00005, dropout_rate = 0.9):
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

        with tf.device("/cpu:0"):
            self.Wemb = self.Wemb = tf.Variable(tf.random_uniform([n_words, word_dim], -0.1, 0.1), dtype=tf.float32,
                                                name='Wemb',trainable=True)  ##without cpu

        #forwardlstm1 = tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=False)
        #backwardlstm11 = tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=False)
        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=False)
        self.lstm1_dropout = tf.contrib.rnn.DropoutWrapper(self.lstm1, output_keep_prob=self.dropout_rate)
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=False)
        self.lstm2_dropout = tf.contrib.rnn.DropoutWrapper(self.lstm2, output_keep_prob=self.dropout_rate)

        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, word_dim], -0.1, 0.1), dtype=tf.float32,
                                          name='encode_image_W', trainable=True)
        self.encode_image_b = tf.Variable(tf.zeros([word_dim], tf.float32), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([lstm_dim, n_words], -0.1, 0.1), dtype=tf.float32,
                                        name='embed_word_W', trainable=True)
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])  ###llj

        caption = tf.placeholder(tf.int32, [self.batch_size,
                                            self.n_caption_lstm_step])  ####llj    make caption start at n_video_lstm_step
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step])  ##llj
        #caption_mask = tf.to_float(tf.not_equal(caption,tf.convert_to_tensor([0]))) #### define mask in function
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
        weight_decay_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) \
                            * self.decay_value
        #loss = loss/tf.reduce_sum(caption_mask) + weight_decay_loss  ###normal loss
        loss = loss / tf.reduce_sum(caption_mask) + weight_decay_loss  #### label smoothing
        return loss, video, caption, caption_mask, probs

    def build_generator(self, beam_size=1, length_normalization_factor=0.5):
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
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
        return video, sentence, probs

        # return video, tf_sentence
    def build_loss(self):###################### change batch_size to 16*batch_size to multisample sentence from multinomial for one video
        video = tf.placeholder(tf.float32, [self.batch_size*8, self.n_video_lstm_step, self.dim_image])  ###llj

        caption = tf.placeholder(tf.int32, [self.batch_size*8,
                                            self.n_caption_lstm_step])  ####llj    make caption start at n_video_lstm_step
        caption_mask = tf.placeholder(tf.float32, [self.batch_size*8, self.n_caption_lstm_step])  ##llj
        # caption_mask = tf.to_float(tf.not_equal(caption,tf.convert_to_tensor([0]))) #### define mask in function
        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W,
                                    self.encode_image_b)  # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size*8, self.n_video_lstm_step,
                                           self.word_dim])  ########potential problem in reshape
        ### add dropout####
        image_emb = tf.layers.dropout(inputs=image_emb, rate=self.dropout_rate)

        state1 = tf.zeros([self.batch_size*8, self.lstm1.state_size], tf.float32)
        state2 = tf.zeros([self.batch_size*8, self.lstm2.state_size], tf.float32)
        padding = tf.zeros([self.batch_size*8, self.word_dim], tf.float32)

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

                    ############################ Decoding Stage ######################################

            for i in range(0, self.n_caption_lstm_step):  ## Phase 2 => only generate captions
                if i == 0:
                    with tf.device("/cpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([self.batch_size*8],
                                                                                  dtype=tf.int64))  ######## embedding begin of sentence <bos>
                else:

                    with tf.device("/cpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,
                                                                          i - 1])  ##without cpu    ### i-1 correspond to the previous word
                #### add dropout##
                current_embed = tf.layers.dropout(inputs=current_embed, rate=self.dropout_rate)

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
        return loss, video, caption, caption_mask

    def build_multinomial_sampler(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_lstm_step, self.word_dim])
        state1 = tf.zeros([self.batch_size, self.lstm1.state_size], tf.float32)
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size], tf.float32)
        padding = tf.zeros([self.batch_size, self.word_dim], tf.float32)

        sampled_words = []
        probs = []
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
                word_probability = tf.nn.log_softmax(logit_words)

                sampled_word = tf.multinomial(word_probability, 1) ### batchsize x 1
                sampled_word = tf.reshape(sampled_word, [-1])
                sampled_words.append(sampled_word)
            sampled_captions = tf.transpose(tf.stack(sampled_words),[1,0])
        return sampled_captions,video


    def build_sampler(self):
        video = tf.placeholder(tf.float32, [None, self.n_video_lstm_step, self.dim_image])
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
                #if i > 0: ### uncomment in training
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

            sampled_captions = tf.transpose(tf.stack(sampled_words),[1,0]) ### batch_size * word_number
        return sampled_captions,video

    def build_mix_sample(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
        caption = tf.placeholder(tf.int32, [self.batch_size,
                                            self.n_caption_lstm_step])
        video_flat = tf.reshape(video, [-1, self.dim_image])
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
        return sampled_captions, video, caption

# =====================================================================================
# Global Parameters
# =====================================================================================

# video_train_caption_file = './data/video_corpus.csv'
# video_test_caption_file = './data/video_corpus.csv'

model_path = './new_multisamp_reinforcement_models'

video_train_feature_file = '/data1/lijun/tensorflow_s2vt/att_5frame_train.txt'

video_test_feature_file = '/data1/lijun/tensorflow_s2vt/att_5frame_test.txt'

#video_train_feature_file = '/home/llj/tensorflow_s2vt/inception_resnet_10frame_train_feature.txt' #### fix multitask cnn, train lstm

#video_test_feature_file = '/home/llj/tensorflow_s2vt/inception_resnet_10frame_val_feature.txt'

video_train_sent_file = '/data1/lijun/tensorflow_s2vt/msvd_sents_train_noval_lc_nopunc.txt'

video_test_sent_file = '/data1/lijun/tensorflow_s2vt/msvd_sents_test_lc_nopunc.txt'

#video_train_feature_file = '/media/llj/storage/all_sentences/msvd_inception_globalpool_train_origin.txt'

#video_test_feature_file = '/media/llj/storage/all_sentences/msvd_inception_globalpool_test_origin.txt'

#video_train_sent_file = '/media/llj/storage/all_sentences/msvd_sents_train_lc_nopunc.txt'

#video_test_sent_file = '/media/llj/storage/all_sentences/msvd_sents_test_lc_nopunc.txt'

#vocabulary_file = '/media/llj/storage/all_sentences/coco_msvd_allvocab.txt'
vocabulary_file = '/data1/lijun/tensorflow_s2vt/msvd_vocabulary1.txt'

model_name = 'reinforce_multisample8_model_att256'
out_file = 'new_multisamp_reinforcement_models/reinforce_multisample8_model_att256.txt'
# =======================================================================================
# Train Parameters
# =======================================================================================
dim_image = 1536
lstm_dim = 1000
word_dim = 500

n_lstm_step = 40
n_caption_lstm_step = 35
n_video_lstm_step = 5

n_epochs = 30
batch_size = 256
#start_learning_rate = 0.001 ### gradient_descent learning rate
start_learning_rate = 0.000001
alpha = 0.05
#caption_mask_out = open('caption_masks.txt', 'w')


def get_video_feature_caption_pair(sent_file=video_train_sent_file, feature_file=video_train_feature_file):
    sents = []
    features = {}
    with open(sent_file, 'r') as video_sent_file:
        for line in video_sent_file:
            line = line.strip()
            id_sent = line.split('\t')
            sents.append((id_sent[0], id_sent[1]))
    with open(feature_file, 'r') as video_feature_file:
        for line in video_feature_file:
            splits = line.split(',')
            id_framenum = splits[0]
            video_id = id_framenum.split('_')[0]
            if video_id not in features:
                features[video_id] = []
            features[video_id].append(splits[1:])
    feature_length = [len(v) for v in features.values()]
    print 'length: ', set(feature_length)
    assert len(set(feature_length)) == 1  ######## make sure the feature lengths are all the same
    sents = np.array(sents)
    return sents, features


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

def get_captions(captions,vid):
    return [y for x,y in captions if x == vid]

def train():  ###### move caption (input_sentence) one column left and also need to move caption_mask (cont_sent)one column left ########################################################llj
    train_captions, train_features = get_video_feature_caption_pair(video_train_sent_file, video_train_feature_file)
    vocabulary = []
    test_captions, test_features = get_video_feature_caption_pair(video_test_sent_file, video_test_feature_file)

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
        bias_init_vector=None)

    _ = model.build_model()
    sampled_captions, multinomial_video_features = model.build_multinomial_sampler()
    #sampled_captions, multinomial_video_features, mix_captions = model.build_mix_sample()
    greedy_captions, greedy_video_features = model.build_sampler()
    rewards = tf.placeholder(tf.float32, [None])
    base_line = tf.placeholder(tf.float32, [None])
    #sampled_caption_mask = tf.placeholder(tf.int32, [None, n_caption_lstm_step]) ### batch_size x n_caption_lstm_step
    loss, loss_features, loss_captions, loss_masks = model.build_loss()

    global_step = tf.Variable(0, trainable=False, name='g_step')
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                               1000, 0.5, staircase=True)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    norm = tf.reduce_sum(loss_masks)
    residual = tf.expand_dims(rewards - base_line,1)
    #sum_loss = -(1-alpha) * tf.reduce_sum(tf.multiply(loss, tf.expand_dims(residual,1)))/norm - alpha * tf.reduce_sum(loss)/norm
    sum_loss = -1 * tf.reduce_sum(tf.multiply(loss, tf.expand_dims(residual, 1))) / norm
    #gvs = optimizer.compute_gradients(sum_loss)
    #capped_gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
    #train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    grads_rl, _ = tf.clip_by_global_norm(tf.gradients(sum_loss, tf.trainable_variables(), aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N), 5)
    grads_and_vars = list(zip(grads_rl,tf.trainable_variables()))
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars,global_step=global_step)


    # config = tf.ConfigProto(allow_soft_placement=True)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth=True
    # config.gpu_options.allocator_type = 'BFC'
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())
    #print 'variables: ',sess.run(tf.global_variables())
    # my tensorflow version is 0.12.1, I write the saver with version 1.0
    saver = tf.train.Saver(max_to_keep=100)


    #optimistic_restore(sess,'/home/llj/tensorflow_s2vt/later_models/third/batch_size16_5later_third_model-9')
    optimistic_restore(sess, '/data1/lijun/tensorflow_s2vt/new_s2vt_models/batch_size64att_new_s2vt_model-10')
    #optimistic_restore(sess, '/home/llj/tensorflow_s2vt/reinforcement_models/reinforce_multisample2_model-6')
    #optimistic_restore(sess, '/home/llj/tensorflow_s2vt/multitask_models/initialize_with_two_model/10batch_size2alpha_0.01_multitask_model-72000')
    #tf.summary.scalar('lr',learning_rate)
    start_t = time.time()

    loss_to_draw = []
    cider_to_draw = []
    #pool = Pool(8)
   # summary_op = tf.summary.merge_all()
#### test restored model first###
    with open(out_file, 'a') as f:
        all_decoded_for_eval = {}
        test_index = list(range(len(test_captions)))
        random.shuffle(test_index)
        ref_decoded = {}
        for aa in xrange(0,len(set(test_captions[:,0])),batch_size):
            id = list(set(test_captions[:,0]))[aa:aa+batch_size]
            test_features_batch = [test_features[x] for x in id]

            feed_dict = {greedy_video_features: test_features_batch}
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
        f.write("METEOR:" + str(scores['METEOR']))
        f.write('\n')
        f.write("metric:" + str(
            1 * scores['CIDEr'] ))
        f.write('\n')


    for epoch in range(0, n_epochs):
        loss_to_draw_epoch = []

        ## randomize the video id order
        index = list(range(len(train_captions)))
        random.shuffle(index)
        ### iterate over the video id
        for start, end in zip(range(0, len(index)-batch_size, batch_size), range(batch_size, len(index), batch_size)):
            ref_decoded = {}
            start_time = time.time()
            vid, sentence = train_captions[index[start:end], 0], train_captions[index[start:end], 1]
            captions_batch = sentence.tolist() #### list batch_size x n_caption_step
            features_batch = [train_features[x] for x in vid]
            # captions_batch = map(lambda x: '<bos> ' + x, captions_batch)
            captions_ind, captions_mask = sentence_padding_toix(captions_batch, wordtoix)

            samples, greedy_words = sess.run([sampled_captions,greedy_captions],feed_dict={
                                                            multinomial_video_features: features_batch,
                                                            greedy_video_features: features_batch})

            temp1_samples = sess.run(sampled_captions,feed_dict={multinomial_video_features: features_batch})
            temp2_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})
            temp3_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})
            temp4_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})
            temp5_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})
            temp6_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})
            temp7_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})
         #   temp8_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})
          #  temp9_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})
          #  temp10_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})
          #  temp11_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})
          #  temp12_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})
          ##  temp13_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})
          #  temp14_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})
           # temp15_samples = sess.run(sampled_captions, feed_dict={multinomial_video_features: features_batch})


            samples = np.vstack((samples,temp1_samples))
            samples = np.vstack((samples, temp2_samples))
            samples = np.vstack((samples, temp3_samples))
            samples = np.vstack((samples, temp4_samples))
            samples = np.vstack((samples, temp5_samples))
            samples = np.vstack((samples, temp6_samples))
            samples = np.vstack((samples, temp7_samples))
            #samples = np.vstack((samples,temp8_samples))
            #samples = np.vstack((samples, temp9_samples))
            #samples = np.vstack((samples, temp10_samples))
           # #samples = np.vstack((samples, temp11_samples))
            #samples = np.vstack((samples, temp12_samples))
            #samples = np.vstack((samples, temp13_samples))
           # samples = np.vstack((samples, temp14_samples))
            #samples = np.vstack((samples, temp15_samples))
            features_batch8 = []
            for i in xrange(8):
                for j in xrange(len(features_batch)):
                    features_batch8.append(features_batch[j])

            mask, multi_decoded = decode_captions_masks(samples,ixtoword)
            greedy_mask, greedy_decoded = decode_captions_masks(greedy_words,ixtoword)
            ###ground truth captions
            ###ground truth captions
            for i, video_id in enumerate(vid):
                ref_decoded[i] = get_captions(train_captions, video_id)
            b = evaluate_captions_cider(ref_decoded,greedy_decoded)
            b = np.vstack((b[:],b[:]))
            b = np.vstack((b[:], b[:]))
            b = np.vstack((b[:], b[:]))
           # b = np.vstack((b[:], b[:]))
            b = b.flatten()

            for i in xrange(1,16):
                for j in xrange(len(vid)):
                    ref_decoded[i*len(vid)+j] = ref_decoded[j]
            #### back up#####
            #ref_decoded = [decode_captions(np.array(captions_ind[j]),ixtoword) for j in range(len(captions_ind))]

            #r = evaluate_captions_cider(ref_decoded,multi_decoded)
            #b = evaluate_captions_cider(ref_decoded,greedy_decoded)
            #################
            r = evaluate_captions_cider(ref_decoded, multi_decoded)
            #b = evaluate_captions_cider(ref_decoded, greedy_decoded)

            #r = evaluate_captions_cider(ref_decoded, greedy_decob = np.vstack((b,b))ded)
            #b = evaluate_captions_cider(ref_decoded, multi_decoded)

            #ref_decoded = [train_captions[train_captions[:, 0] == x, 1].tolist() for x in vid]
           # r = [evaluate_captions([k],[v]) for k,v in zip(ref_decoded, multi_decoded)]
            #r =pool.map(evaluate_captions_wrapper, zip(ref_decoded, multi_decoded))
            #b = [evaluate_captions([k],[v]) for k,v in zip(ref_decoded, greedy_decoded)]
            #b = pool.map(evaluate_captions_wrapper, zip(ref_decoded, greedy_decoded))
            #loss_to_draw_epoch.extend(b)
            #print 'r-b: ', map(operator.sub, r, b)
            #print 'r-b: ',np.mean(np.array(r) - np.array(b))
            #############back up#############
            #feed_dict = {loss_masks: mask, loss_captions: samples, loss_features: features_batch, rewards: r, base_line: b}
            ###################
            feed_dict = {loss_masks: mask, loss_captions: samples, loss_features: features_batch8, rewards: r,
                         base_line: b}

            _,loss_val = sess.run([train_op, sum_loss],feed_dict)
            print 'idx: ', start, ' rate: ', sess.run(learning_rate)," Epoch: ", epoch, " loss: ", loss_val,\
                ' Elapsed time: ', str((time.time() - start_time))
            loss_to_draw_epoch.append(loss_val)

        with open(out_file,'a') as f:
            all_decoded_for_eval = {}
            test_index = list(range(len(test_captions)))
            random.shuffle(test_index)
            ref_decoded = {}
            for aa in xrange(0, len(set(test_captions[:, 0])), batch_size):
                id = list(set(test_captions[:, 0]))[aa:aa + batch_size]
                test_features_batch = [test_features[x] for x in id]

                feed_dict = {greedy_video_features: test_features_batch}
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
        cider_to_draw.append([scores['CIDEr']])

        if np.mod(epoch, 1) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, model_name), global_step=epoch)

        # draw loss curve every epoch
        loss_to_draw.append(np.mean(loss_to_draw_epoch))



def test(model_path='/home/llj/tensorflow_s2vt/reinforcement_models/'):
    test_captions, test_features = get_video_feature_caption_pair(video_test_sent_file, video_test_feature_file)

    ixtoword = pd.Series(np.load('./data/ixtoword.npy').tolist())

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
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)

    for i in xrange(3):
        model_path_last = model_path + 'vocab1_model-' + str(i)
        out_file = 'meteor_reinforcement_batchsize_'+str(batch_size)+'_ep'+ str(i) + '.txt'
        saver = tf.train.Saver()
        saver.restore(sess, model_path_last)

        test_output_txt_fd = open(out_file, 'w')
        for key, values in test_features.iteritems():
            generated_word_index = sess.run(captions_tf, feed_dict={video_tf: [test_features[key]]})
            generated_words = ixtoword[generated_word_index]

            punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
            generated_words = generated_words[:punctuation]

            generated_sentence = ' '.join(generated_words)
            generated_sentence = generated_sentence.replace('<bos> ', '')
            generated_sentence = generated_sentence.replace(' <eos>', '')
            print generated_sentence, '\n'
            test_output_txt_fd.write(key + '\t')
            test_output_txt_fd.write(generated_sentence + '\n')

def evaluation(model_path='/data1/lijun/tensorflow_s2vt/new_multisamp_reinforcement_models/'):
    test_captions, test_features = get_video_feature_caption_pair(video_test_sent_file, video_test_feature_file)

    ixtoword = pd.Series(np.load('./new_vocab1_data/ixtoword.npy').tolist())
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.InteractiveSession(config=config)

    #model_path_last = model_path + 'reinforce_mix-2'
    #model_path_last = '/home/llj/tensorflow_s2vt/reinforcement_models/reinforce_multisample_model-19'

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

    with open('new_multisamp_reinforcement_models/reinforce_multisample8_test.txt', 'a') as f:
      for i in xrange(9,18,1):
        start_time = time.time()
        model_path_last = model_path + 'reinforce_multisample8_model-' + str(i)
        saver.restore(sess, model_path_last)
        all_decoded_for_eval = {}
        test_index = list(range(len(test_captions)))
        random.shuffle(test_index)
        ref_decoded = {}
        for aa in xrange(0, len(set(test_captions[:, 0])), batch_size):

            id = list(set(test_captions[:, 0]))[aa:aa + batch_size]
            test_video_batch = [test_features[x] for x in id]

            feed_dict = {greedy_video_features: test_video_batch}
            greedy_words = sess.run(greedy_captions, feed_dict)  #### batch_size x num of each words
            greedy_decoded = decode_captions(np.array(greedy_words), ixtoword)
            for videoid in id:
                if videoid not in all_decoded_for_eval:
                    all_decoded_for_eval[videoid] = []

            [all_decoded_for_eval[x].append(y) for x, y in zip(id, greedy_decoded)]
        end_time = time.time()
        print('test generate time : ',end_time-start_time)

        for num in xrange(0, len(test_captions), batch_size):

            videoid = test_captions[num:num + batch_size, 0]
            for id in videoid:
                if id not in ref_decoded:
                    ref_decoded[id] = []
            [ref_decoded[x].append(y) for x, y in zip(videoid, test_captions[num:num + batch_size, 1])]

        scores = evaluate_for_particular_captions(all_decoded_for_eval, ref_decoded)

	f.write("Epoch : " + str(i))
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
