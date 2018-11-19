import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
import cv2
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math
from beam_search import *
from inception_resnet_v2 import *
import glob
from evaluation import *
import multiprocessing
import time

slim = tf.contrib.slim

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
                        default='evaluate', type=str)
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
                 width = 299, height = 299, channels= 3):
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
        ########## inception resnet v2####
        ###preprocessing###
        video_frames = tf.placeholder(tf.float32,[self.batch_size, self.n_video_lstm_step, self.height, self.width, self.channels])
        all_frames = tf.reshape(video_frames,[-1, self.height, self.width, self.channels])

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
        weight_decay_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' or 'BatchNorm' not in v.name]) \
                            * self.decay_value
        #loss = loss/tf.reduce_sum(caption_mask) + weight_decay_loss  ###normal loss
        loss = loss / tf.reduce_sum(caption_mask) + weight_decay_loss  #### label smoothing
        return loss, video_frames, caption, caption_mask, probs

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
                    #tf.get_variable_scope().reuse_variables() ### uncomment in training
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
                if i > 0:  ### comment when training
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

# =====================================================================================
# Global Parameters
# =====================================================================================

# video_train_caption_file = './data/video_corpus.csv'
# video_test_caption_file = './data/video_corpus.csv'

#model_path = './vocab1_models'
model_path = './new_e2e_models'

video_path = '/data1/lijun/msvd/video-frames'

#video_train_feature_file = '/media/llj/storage/all_sentences/msvd_inception_globalpool_train_origin.txt'

#video_test_feature_file = '/media/llj/storage/all_sentences/msvd_inception_globalpool_test_origin.txt'

video_train_sent_file = '/data1/lijun/tensorflow_s2vt/msvd_sents_train_noval_lc_nopunc.txt'

#video_test_sent_file = '/data1/lijun/tensorflow_s2vt/msvd_sents_test_lc_nopunc.txt'

video_test_sent_file = '/data1/lijun/tensorflow_s2vt/msvd_all.txt'
#vocabulary_file = '/media/llj/storage/all_sentences/coco_msvd_allvocab.txt'
vocabulary_file = '/data1/lijun/tensorflow_s2vt/msvd_vocabulary1.txt'
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
batch_size = 16 
start_learning_rate = 0.00001
width = 299
height = 299
#caption_mask_out = open('caption_masks.txt', 'w')


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
    for x in vid:
        if len(video_frames[x]) == 4:
            print('x : ',x)
    print 'length: ', set(feature_length)
    sys.stdout.flush()
    assert len(set(feature_length)) == 1  ######## make sure the feature lengths are all the same
    sents = np.array(sents)
    return sents, video_frames


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


def train():  ###### move caption (input_sentence) one column left and also need to move caption_mask (cont_sent)one column left ########################################################llj
    train_captions, video_frames = get_video_feature_caption_pair(video_train_sent_file, video_path, num_frame_per_video=n_video_lstm_step)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    vocabulary = []
    test_captions, test_video_frames = get_video_feature_caption_pair(video_test_sent_file, video_path, num_frame_per_video=n_video_lstm_step)

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

    tf_loss, tf_video, tf_caption, tf_caption_mask, tf_probs = model.build_model()
    # config = tf.ConfigProto(allow_soft_placement=True)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    # config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)

    # my tensorflow version is 0.12.1, I write the saver with version 1.0
    saver = tf.train.Saver(max_to_keep=100, write_version=1)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                               40000, 0.5, staircase=True)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #gradients, variables = zip(*optimizer.compute_gradients(tf_loss))
    #gradients, _ = tf.clip_by_global_norm(gradients, 10)
    #train_op = optimizer.apply_gradients(zip(gradients, variables),global_step=global_step)

    ###################clipping every gradient ################
    #gvs = optimizer.compute_gradients(tf_loss, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    #capped_gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
    #train_op = optimizer.apply_gradients(capped_gvs,global_step=global_step)
    ############## clipping every gradient####
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(tf_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 10)
    train_op = optimizer.apply_gradients(zip(gradients, variables),global_step=global_step)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss, global_step=global_step)
    tf.global_variables_initializer().run()
    #tf.summary.scalar('lr',learning_rate)
    
    #### origin load ###
    optimistic_restore(sess,'inception_resnet_v2_2016_08_30.ckpt')
    #optimistic_restore(sess, '/data1/lijun/tensorflow_s2vt/new_s2vt_models/batch_size64_new_s2vt_model-22')
    ############## llj   #################################################################################################

    #optimistic_restore(sess, '/home/llj/tensorflow_s2vt/e2e_models/10batch_size2cnn_model_false_withval-15000')
    # new_saver = tf.train.Saver()
    # new_saver = tf.train.import_meta_graph('./rgb_models/model-1000.meta')
    # new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))

    loss_fd = open('vocab_loss.txt', 'w')
    loss_to_draw = []
   # summary_op = tf.summary.merge_all()
    ############# test before training########

    greedy_captions, greedy_video_features = model.build_sampler()
    ######


    with open('new_e2e_models/noinit_e2e_batch16_lr1e-5_train_withoutval.txt', 'a') as f:
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
        f.write("METEOR:" + str(scores['METEOR']))
        f.write('\n')
        f.write("metric:" + str(
            1 * scores['METEOR'] ))
        f.write('\n')

    iter_count = 0
    for epoch in range(0, n_epochs):

        loss_to_draw_epoch = []

        ## randomize the video id order
        index = list(range(len(train_captions)))
        random.shuffle(index)
        ### iterate over the video id
        for start, end in zip(range(0, len(index) - batch_size, batch_size), range(batch_size, len(index), batch_size)):
            iter_count += 1
            start_time = time.time()
            vid, sentence = train_captions[index[start:end], 0], train_captions[index[start:end], 1]
            captions_batch = sentence.tolist()
            video_frames_batch = [video_frames[x] for x in vid]
            video_batch = image_reading_processing(video_frames_batch)
            # captions_batch = map(lambda x: '<bos> ' + x, captions_batch)
            captions_ind, captions_mask = sentence_padding_toix(captions_batch, wordtoix)

            _, loss_val = sess.run(
                [train_op, tf_loss],
                feed_dict={
                    tf_video: video_batch,
                    tf_caption: captions_ind,
                    tf_caption_mask: captions_mask
                })
            loss_to_draw_epoch.append(loss_val)

            if np.mod(iter_count, 3000) == 0:
                with open('new_e2e_models/noinit_e2e_batch16_lr1e-5_train_withoutval.txt', 'a') as f:
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
                print 'CIDEr: ',scores['CIDEr']

                loss_to_draw.append(np.mean(loss_to_draw_epoch))
                loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')

            if np.mod(iter_count, 3000) == 0:
                print "iter_count ", iter_count, " is done. Saving the model ..."
                saver.save(sess, os.path.join(model_path,
                                              str(n_video_lstm_step) + 'batch_size' + str(batch_size) + 'noinit_e2e_batch32_lr1e-5_train_withoutval'),
                           global_step=iter_count)
            print 'idx: ', start, ' rate: ', sess.run(learning_rate)," Epoch: ", epoch, " loss: ", loss_val,\
                ' Elapsed time: ', str((time.time() - start_time))



        # draw loss curve every epoch


    loss_fd.close()


def test(model_path='/home/llj/tensorflow_s2vt/vocab1_models/'):
    test_captions, video_frames = get_video_feature_caption_pair(video_test_sent_file, video_path, num_frame_per_video=n_video_lstm_step)

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

    config = tf.ConfigProto(allow_growth=True, allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
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

def evaluation(model_path='/data1/lijun/tensorflow_s2vt/new_e2e_models/'):
    
    generate_start = time.time()
    test_captions, test_video_frames = get_video_feature_caption_pair(video_test_sent_file, video_path,
                                                                      num_frame_per_video=n_video_lstm_step)

    ixtoword = pd.Series(np.load('./new_vocab1_data/ixtoword.npy').tolist())
    model_start = time.time()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.InteractiveSession(config=config)

    model_path_last = model_path + '10batch_size2cnn_model_false-24000'


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

    with open('new_e2e_models/batch4_lr1e-5_test_all.txt', 'w') as f:
      for i in xrange(42000,50000,36000):
        model_path_last = model_path + '5batch_size16noinit_e2e_batch32_lr1e-5_train_withoutval-'+str(i)
        saver.restore(sess, model_path_last)
        model_end = time.time() - model_start
        print('model initial time : ', model_end)
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
        generate_end = time.time() - generate_start
        print('generate time : ', generate_end)

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
            1 * scores['METEOR'] ))
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
