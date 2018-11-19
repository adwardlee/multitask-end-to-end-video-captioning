import matplotlib
matplotlib.use('Agg') 
from scipy import io


import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import pdb
import time
import json
from collections import defaultdict

import time
import cv2
import argparse
import matplotlib.pyplot as plt
import random
import math
from beam_search import *
import glob
from cider_evaluation import *
import psutil


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

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_video_lstm_steps, n_caption_lstm_steps, drop_out_rate, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_video_lstm_steps = n_video_lstm_steps
	self.n_caption_lstm_steps = n_caption_lstm_steps
        self.drop_out_rate = drop_out_rate

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1,seed=seed_num), name='Wemb')

        self.lstm3 = tf.contrib.rnn.BasicLSTMCell(self.dim_hidden, state_is_tuple=False)
        self.lstm3_dropout = tf.contrib.rnn.DropoutWrapper(self.lstm3,output_keep_prob = self.drop_out_rate)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1,seed=seed_num), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')


        self.embed_att_w = tf.Variable(tf.random_uniform([dim_hidden, 1], -0.1,0.1,seed=seed_num), name='embed_att_w')
        self.embed_att_Wa = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1,0.1,seed=seed_num), name='embed_att_Wa')
        self.embed_att_Ua = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden],-0.1,0.1,seed=seed_num), name='embed_att_Ua')
        self.embed_att_ba = tf.Variable( tf.zeros([dim_hidden]), name='embed_att_ba')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1,seed=seed_num), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        self.embed_nn_Wp = tf.Variable(tf.random_uniform([3*dim_hidden, dim_hidden], -0.1,0.1,seed=seed_num), name='embed_nn_Wp')
        self.embed_nn_bp = tf.Variable(tf.zeros([dim_hidden]), name='embed_nn_bp')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_steps, self.dim_image]) # b x n x d
        #video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_steps]) # b x n

        caption = tf.placeholder(tf.int32, [self.batch_size, n_caption_lstm_steps]) # b x 16
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, n_caption_lstm_steps]) # b x 16

        video_flat = tf.reshape(video, [-1, self.dim_image]) # (b x n) x d
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (b x n) x h
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_lstm_steps, self.dim_hidden]) # b x n x h
        image_emb = tf.transpose(image_emb, [1,0,2]) # n x b x h

        state1 = tf.zeros([self.batch_size, self.lstm3.state_size]) # b x s
        h_prev = tf.zeros([self.batch_size, self.dim_hidden]) # b x h

        loss_caption = 0.0

        current_embed = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_video_lstm_steps,1,1]) # n x h x 1
        image_part = tf.matmul(image_emb, tf.tile(tf.expand_dims(self.embed_att_Ua, 0), [self.n_video_lstm_steps,1,1])) + self.embed_att_ba # n x b x h

        with tf.variable_scope("s2vt") as scope:

          for i in range(n_caption_lstm_steps):
            if i > 0: tf.get_variable_scope().reuse_variables()
            e = tf.tanh(tf.matmul(h_prev, self.embed_att_Wa) + image_part) # n x b x h
            e = tf.matmul(e, brcst_w)    # unnormalized relevance score 
            e = tf.reduce_sum(e,2) # n x b
            e_hat_exp = tf.exp(e)
           # e_hat_exp = tf.mul(tf.transpose(video_mask), tf.exp(e)) # n x b 
            denomin = tf.reduce_sum(e_hat_exp,0) # b
            denomin = denomin + tf.to_float(tf.equal(denomin, 0))   # regularize denominator
            alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h  # normalize to obtain alpha
            temp_alphas = tf.div(e_hat_exp,denomin) ### n x b
            temp_alphas = tf.transpose(temp_alphas,[1,0]) #### b x n
            alphas_1 = temp_alphas[:,0:8]  # first quarter b x 8
            alphas_2 = temp_alphas[:,8:16]
            alphas_3 = temp_alphas[:,16:24]
            alphas_4 = temp_alphas[:,24:32] ## last quarter
            attention_list = tf.multiply(alphas, image_emb) # n x b x h
            atten = tf.reduce_sum(attention_list,0) # b x h       #  soft-attention weighted sum
            

            with tf.variable_scope("LSTM3"):
                output1, state1 = self.lstm3_dropout( tf.concat([atten, current_embed],axis=1), state1 ) # b x h

            output2 = tf.tanh(tf.nn.xw_plus_b(tf.concat([output1,atten,current_embed],axis=1), self.embed_nn_Wp, self.embed_nn_bp)) # b x h
            h_prev = output1 # b x h
            labels = tf.expand_dims(caption[:,i], 1) # b x 1
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
            concated = tf.concat([indices, labels],axis=1) # b x 2
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0) # b x w
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i])

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b) # b x w
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit_words, labels = onehot_labels) # b x 1
            #regularizer = beta * tf.maximum(0.0,m-tf.reduce_sum(alphas_1,axis=1)+tf.reduce_sum(alphas_4,axis=1)) * caption_mask[:,i]# b x 1
            regularizer = beta * tf.maximum(0.0,m-tf.reduce_sum(alphas_1,axis=1)) * caption_mask[:,i]# b x 1
            cross_entropy = cross_entropy * caption_mask[:,i] +  regularizer ####add the regularizer
            loss_caption += tf.reduce_sum(cross_entropy) # 1

        loss_caption = loss_caption / tf.reduce_sum(caption_mask)
        loss = loss_caption
        return loss, video, caption, caption_mask


    def build_generator(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_steps, self.dim_image])
        #video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_lstm_steps, self.dim_hidden])
        image_emb = tf.transpose(image_emb, [1,0,2])

        state1 = tf.zeros([self.batch_size, self.lstm3.state_size])
        h_prev = tf.zeros([self.batch_size, self.dim_hidden])

        generated_words = []

        current_embed = tf.zeros([self.batch_size, self.dim_hidden])
        brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_video_lstm_steps,1,1])   # n x h x 1
        image_part = tf.matmul(image_emb, tf.tile(tf.expand_dims(self.embed_att_Ua, 0), [self.n_video_lstm_steps,1,1])) +  self.embed_att_ba # n x b x h
        with tf.variable_scope("s2vt") as scope:

          for i in range(n_caption_lstm_steps):
            if i > 0: tf.get_variable_scope().reuse_variables()
            e = tf.tanh(tf.matmul(h_prev, self.embed_att_Wa) + image_part) # n x b x h
            e = tf.matmul(e, brcst_w)
            e = tf.reduce_sum(e,2) # n x b
            e_hat_exp = tf.exp(e)
            #e_hat_exp = tf.mul(tf.transpose(video_mask), tf.exp(e)) # n x b
            denomin = tf.reduce_sum(e_hat_exp,0) # b
            denomin = denomin + tf.to_float(tf.equal(denomin, 0))
            alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h
            attention_list = tf.multiply(alphas, image_emb) # n x b x h                
            atten = tf.reduce_sum(attention_list,0) # b x h

            with tf.variable_scope("LSTM3") as vs:
                output1, state1 = self.lstm3( tf.concat([atten, current_embed],axis=1), state1 ) # b x h

            output2 = tf.tanh(tf.nn.xw_plus_b(tf.concat([output1,atten,current_embed],axis=1), self.embed_nn_Wp, self.embed_nn_bp)) # b x h
            h_prev = output1
            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b) # b x w
            max_prob_index = tf.argmax(logit_words, 1) # b
            generated_words.append(max_prob_index) # b
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)

        generated_words = tf.transpose(tf.stack(generated_words))
        return video, generated_words

    def build_sampler(self):
	saved_alphas = []
        video = tf.placeholder(tf.float32, [None, self.n_video_lstm_steps, self.dim_image])
        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        #image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_lstm_step, self.word_dim])
        #state1 = tf.zeros([self.batch_size, self.lstm1.state_size], tf.float32)
        #state2 = tf.zeros([self.batch_size, self.lstm2.state_size], tf.float32)
        #padding = tf.zeros([self.batch_size, self.word_dim], tf.float32)
        image_emb = tf.reshape(image_emb, [-1, self.n_video_lstm_steps, self.dim_hidden])
        image_emb = tf.transpose(image_emb, [1,0,2])

        state1 = tf.zeros(tf.stack([tf.shape(video)[0], self.lstm3.state_size]), tf.float32)
        h_prev = tf.zeros(tf.stack([tf.shape(video)[0], self.dim_hidden]), tf.float32)

        generated_words = []
        current_embed = tf.zeros(tf.stack([tf.shape(video)[0], self.dim_hidden]),tf.float32)
        brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_video_lstm_steps,1,1])   # n x h x 1
        image_part = tf.matmul(image_emb, tf.tile(tf.expand_dims(self.embed_att_Ua, 0), [self.n_video_lstm_steps,1,1])) +  self.embed_att_ba # n x b x h
        with tf.variable_scope("s2vt") as scope:

          for i in range(n_caption_lstm_steps):
            if i > 0: 
                tf.get_variable_scope().reuse_variables()
            e = tf.tanh(tf.matmul(h_prev, self.embed_att_Wa) + image_part) # n x b x h
            e = tf.matmul(e, brcst_w)
            e = tf.reduce_sum(e,2) # n x b
            e_hat_exp = tf.exp(e)
            #e_hat_exp = tf.mul(tf.transpose(video_mask), tf.exp(e)) # n x b
            denomin = tf.reduce_sum(e_hat_exp,0) # b
            denomin = denomin + tf.to_float(tf.equal(denomin, 0))
            alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h
            attention_list = tf.multiply(alphas, image_emb) # n x b x h                
            atten = tf.reduce_sum(attention_list,0) # b x n

############### saved alphas ##############
            saved_alphas.append(tf.div(e_hat_exp,denomin)) # n x b
######################################################
            with tf.variable_scope("LSTM3") as vs:
                output1, state1 = self.lstm3( tf.concat([atten, current_embed],axis=1), state1 ) # b x h

            output2 = tf.tanh(tf.nn.xw_plus_b(tf.concat([output1,atten,current_embed],axis=1), self.embed_nn_Wp, self.embed_nn_bp)) # b x h
            h_prev = output1
            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b) # b x w
            max_prob_index = tf.argmax(logit_words, 1) # b
            generated_words.append(max_prob_index) # b
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)

            sampled_captions = tf.transpose(tf.stack(generated_words),[1,0])
        return sampled_captions,video, saved_alphas ##### alphas : n_caption_steps x n x b

# =====================================================================================
# Global Parameters
# =====================================================================================

# video_train_caption_file = './data/video_corpus.csv'
# video_test_caption_file = './data/video_corpus.csv'

#model_path = '/home/lijun/tensor_examples/res_models1'

model_path = './attention_models'

#video_train_feature_file = '/home/lijun/tensor_examples/tf_inceptionres_v2_train_noval_feature.txt'

#video_test_feature_file = '/home/lijun/tensor_examples/tf_inceptionres_v2_test_feature.txt'

#video_train_feature_file = '/home/crcv/tensorflow/models/5secondhalf_feature_train.txt' ############################ change ################

video_train_feature_file = '/data1/lijun/tensorflow_s2vt/5frame_train.txt'

video_test_feature_file = '/data1/lijun/tensorflow_s2vt/5frame_all.txt'

#video_test_feature_file = '/home/crcv/tensorflow/models/5secondhalf_feature_val.txt' ###################### change ##################

#video_train_feature_file = '/home/lijun/tensor_examples/train_25_feature_inception_resnetv2'

#video_test_feature_file = '/home/lijun/tensor_examples/test_25_feature_inception_resnetv2'

video_train_sent_file = '/data1/lijun/tensorflow_s2vt/msvd_sents_train_noval_lc_nopunc.txt'

video_test_sent_file = '/data1/lijun/tensorflow_s2vt/msvd_all.txt'

#vocabulary_file = '/home/lijun/tensor_examples/coco_msvd_allvocab.txt'
vocabulary_file = '/data1/lijun/tensorflow_s2vt/msvd_vocabulary1.txt'

model_name = '_beta10_m05_32img_attention_model'
cider_img_name = '32img_attention_cider'
loss_img_name = '32img_attention_cider_loss'
out_file = 'beta10_m05_batch64_32img_attention_model_val'
# =======================================================================================
# Train Parameters
# =======================================================================================
#dim_image = 1024
dim_image = 1536
dim_hidden = 1000
word_dim = 500

m = 0.5
beta = 10 ##### 0.5  no 0.05
seed_num = 16#### 2,4,8,16,32,64,128

n_lstm_step = 40
n_caption_lstm_steps = 35
n_video_lstm_steps = 5 ####################### change ####################

n_epochs = 20
batch_size = 1
start_learning_rate = 0.0001
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
        one_caption_mask = np.ones(n_caption_lstm_steps)
        word = each_cap.lower().split(' ')
        if len(word) < n_caption_lstm_steps:
            for i in range(len(word), n_caption_lstm_steps):
                captions_batch[idx] = captions_batch[idx] + ' <eos>'
                if i != len(word):
                    one_caption_mask[i] = 0
        else:
            new_word = ''
            for i in range(n_caption_lstm_steps - 1):
                new_word = new_word + word[i] + ' '
            captions_batch[idx] = new_word + '<eos>'
        # one_caption_mask=np.reshape(one_caption_mask,(-1,n_caption_lstm_step))
        captions_mask.append(one_caption_mask)
    captions_mask = np.reshape(captions_mask, (-1, n_caption_lstm_steps))
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


def train():  ###### move caption (input_sentence) one column left and also need to move caption_mask (cont_sent)one column left ########################################################llj
    train_captions, train_features = get_video_feature_caption_pair(video_train_sent_file, video_train_feature_file)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
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
        dim_hidden=dim_hidden,
        batch_size=batch_size,
        n_video_lstm_steps=n_video_lstm_steps,
        n_caption_lstm_steps=n_caption_lstm_steps,
	drop_out_rate=0.9,
        bias_init_vector=None)

    tf_loss, tf_video, tf_caption, tf_caption_mask = model.build_model()
    # config = tf.ConfigProto(allow_soft_placement=True)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.InteractiveSession(config=config)

    # my tensorflow version is 0.12.1, I write the saver with version 1.0
    saver = tf.train.Saver(max_to_keep=100)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                               10000, 0.5, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(tf_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 10)
    train_op = optimizer.apply_gradients(zip(gradients, variables),global_step=global_step)
    #gvs = optimizer.compute_gradients(tf_loss)
    #capped_gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
    #train_op = optimizer.apply_gradients(capped_gvs,global_step=global_step)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss, global_step=global_step)
    tf.global_variables_initializer().run()
    #tf.summary.scalar('lr',learning_rate)

    # new_saver = tf.train.Saver()
    # new_saver = tf.train.import_meta_graph('./rgb_models/model-1000.meta')
    # new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))

    loss_fd = open('batch_size'+str(batch_size)+'loss.txt', 'w')
    loss_to_draw = []
    cider_to_draw = []
   # summary_op = tf.summary.merge_all()
    greedy_captions, greedy_video_features,_ = model.build_sampler()
    random.seed(seed_num)
    for epoch in range(0, n_epochs):
        loss_to_draw_epoch = []

        ## randomize the video id order
        index = list(range(len(train_captions)))
        random.shuffle(index)
        ### iterate over the video id
        for start, end in zip(range(0, len(index) - batch_size, batch_size), range(batch_size, len(index), batch_size)):
            start_time = time.time()
            vid, sentence = train_captions[index[start:end], 0], train_captions[index[start:end], 1]
            captions_batch = sentence.tolist()
            features_batch = [train_features[x] for x in vid]
            # captions_batch = map(lambda x: '<bos> ' + x, captions_batch)
            captions_ind, captions_mask = sentence_padding_toix(captions_batch, wordtoix)

            _, loss_val = sess.run(
                [train_op, tf_loss],
                feed_dict={
                    tf_video: features_batch,
                    tf_caption: captions_ind,
                    tf_caption_mask: captions_mask
                })
            loss_to_draw_epoch.append(loss_val)

            print 'idx: ', start, ' rate: ', sess.run(learning_rate)," Epoch: ", epoch, " loss: ", loss_val,\
                ' Elapsed time: ', str((time.time() - start_time))
            loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')

        with open(out_file, 'a') as f: ####################### change #############################################
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

            for num in xrange(0, len(test_captions), batch_size):

                videoid = test_captions[num:num + batch_size, 0]
                for id in videoid:
                    if id not in ref_decoded:
                        ref_decoded[id] = []
                [ref_decoded[x].append(y) for x, y in zip(videoid, test_captions[num:num + batch_size, 1])]

            scores = evaluate_for_particular_captions(all_decoded_for_eval, ref_decoded)

            f.write('Epoch %d\n' % epoch)
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
                1 * scores['CIDEr']))
            f.write('\n')
        print 'CIDEr: ', scores['CIDEr']

	cider_to_draw.append([scores['CIDEr']])

        # draw loss curve every epoch
        loss_to_draw.append(np.mean(loss_to_draw_epoch))

        if np.mod(epoch, 1) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'batch_size'+ str(batch_size) + model_name), global_step=epoch) ################ change ###############

    loss_fd.close()


def test(model_path='/home/lijun/tensor_examples/models/'):
    test_captions, test_features = get_video_feature_caption_pair(video_test_sent_file, video_test_feature_file)

    ixtoword = pd.Series(np.load('/home/lijun/tensor_examples/data/ixtoword.npy').tolist())

    model = Video_Caption_Generator(
        dim_image=dim_image,
        n_words=len(ixtoword),
        dim_hidden=dim_hidden,
        batch_size=batch_size,
        n_video_lstm_steps=n_video_lstm_steps,
        n_caption_lstm_steps=n_caption_lstm_steps,
	drop_out_rate=1,
        bias_init_vector=None)

    # video_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    video_tf, captions_tf, logprob_tf = model.build_generator()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.InteractiveSession(config=config)

    for i in xrange(15):
        model_path_last = model_path + 'batch_size' + str(batch_size) + 'model-' + str(i)
        out_file = 'global_clipping_adam_vocab51915_batchsize_'+str(batch_size)+'_ep'+ str(i) + '.txt'
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

def evaluation(model_path='/data1/lijun/tensorflow_s2vt/attention_models/'):
    start_time = time.time()
    test_captions, test_features = get_video_feature_caption_pair(video_test_sent_file, video_test_feature_file)

    ixtoword = pd.Series(np.load('./new_vocab1_data/ixtoword.npy').tolist())
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    #model_path_last = model_path + 'batch_size16_50dpp_model-14'


    model = Video_Caption_Generator(
        dim_image=dim_image,
        n_words=len(ixtoword),
        dim_hidden=dim_hidden,
        batch_size=batch_size,
        n_video_lstm_steps=n_video_lstm_steps,
        n_caption_lstm_steps=n_caption_lstm_steps,
	drop_out_rate=1,
        bias_init_vector=None)
    greedy_captions, greedy_video_features,saved_alphas = model.build_sampler()

    saver = tf.train.Saver()


    



    #saver.restore(sess, model_path_last)

    with open('attention_models/32img_attention_beta10_m05_test.txt', 'a') as f:
      for i in xrange(16,17,1):
        model_path_last = model_path + 'batch_size64_beta10_m05_32img_attention_model-' + str(i)
        saver.restore(sess,model_path_last)
        all_decoded_for_eval = {}
        test_index = list(range(len(test_captions)))
        random.shuffle(test_index)
        ref_decoded = {}
        process = psutil.Process(os.getpid())
        print('process memory : ', process.memory_info().rss)

######################### attention ####
        attention = []
        matlab_att = []
        words_valid = []
        for aa in xrange(0, len(set(test_captions[:, 0])), batch_size):

            id = list(set(test_captions[:, 0]))[aa:aa + batch_size]
            test_video_batch = [test_features[x] for x in id]

            feed_dict = {greedy_video_features: test_video_batch}
            greedy_words,batch_alphas = sess.run([greedy_captions,saved_alphas], feed_dict)  #### batch_size x num of each words

            ###### batch_alphas : n_caption_steps x n x b...
            batch_alphas = np.transpose(batch_alphas,(2,0,1)) ########## b x n_caption_steps x n
            masks, greedy_decoded = decode_captions_masks(np.array(greedy_words), ixtoword)  #@@@@@ b x n_caption_steps
            for videoid in id:
                if videoid not in all_decoded_for_eval:
                    all_decoded_for_eval[videoid] = []

            [all_decoded_for_eval[x].append(y) for x, y in zip(id, greedy_decoded)]

        end_time = time.time() - start_time
        print('generation time : ', end_time)

        for num in xrange(0, len(test_captions), batch_size):

            videoid = test_captions[num:num + batch_size, 0]
            for id in videoid:
                if id not in ref_decoded:
                    ref_decoded[id] = []
            [ref_decoded[x].append(y) for x, y in zip(videoid, test_captions[num:num + batch_size, 1])]

        scores = evaluate_for_particular_captions(all_decoded_for_eval, ref_decoded)

	f.write('\n')
	f.write('epoch: '+str(i))
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
