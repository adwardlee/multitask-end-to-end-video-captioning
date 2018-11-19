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
import glob
from evaluation import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, word_dim, lstm_dim, batch_size, n_lstm_steps, n_video_lstm_step,
                 n_caption_lstm_step, bias_init_vector=None, beam_size = 1):
        self.dim_image = dim_image
        self.n_words = n_words
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps  #### number of lstm cell
        self.n_video_lstm_step = n_video_lstm_step  ### frame number
        self.n_caption_lstm_step = n_caption_lstm_step  #### caption number
        self.beam_size = beam_size

        with tf.device("/cpu:0"):
            self.Wemb = self.Wemb = tf.Variable(tf.random_uniform([n_words, word_dim], -0.1, 0.1), dtype=tf.float32,
                                                name='Wemb')  ##without cpu
        # self.bemb = tf.Variable(tf.zeros([dim_hidden]), name='bemb')

        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=False)
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=False)

        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, word_dim], -0.1, 0.1), dtype=tf.float32,
                                          name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([word_dim], tf.float32), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([lstm_dim, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])  ###llj

        caption = tf.placeholder(tf.int32, [self.batch_size,
                                            self.n_caption_lstm_step])  ####llj    make caption start at n_video_lstm_step
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step])  ##llj

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W,
                                    self.encode_image_b)  # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_lstm_step,
                                           self.word_dim])  ########potential problem in reshape

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
                    output1, state1 = self.lstm1(image_emb[:, i, :], state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([output1, padding], 1), state2)

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
                tf.get_variable_scope().reuse_variables()
                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding, state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([output1, tf.to_float(current_embed)], 1), state2)

                    # labels = tf.expand_dims(caption[:, i], 1)#### batch_size x 1 ####### i correspond to current word
                    # indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) ###### batch_size x 1
                    # concated = tf.concat([indices, labels],1)  #### make indices and labels pair batchsize x 2
                # onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words],axis=0), 1.0, 0.0) ##### batch_size number of one hot word vector### batch_size x n_words

                labels = tf.convert_to_tensor(caption[:, i])  #### batch_size x 1 ####### i correspond to current word
                onehot_labels = tf.one_hot(labels, self.n_words, 1.0, 0.0)

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_words)
                cross_entropy = cross_entropy * caption_mask[:,i]  ######### need to move caption_mask (cont_sent)one column left ##########################llj
                probs.append(logit_words)

                current_loss = tf.reduce_sum(cross_entropy) / self.batch_size
                loss = loss + current_loss
        return loss, video, caption, caption_mask, probs

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
        return sampled_captions,video


    #def beam_probability(self, sess, state_feed, input_feed, state1, beam_size):
    def beam_probability(self):
        state2_feed = tf.placeholder(tf.float32, [1,self.lstm2.state_size])
        state1_feed = tf.placeholder(tf.float32, [1,self.lstm1.state_size])
        input_feed = tf.placeholder(tf.int32, [1])
        padding = tf.zeros([1, self.word_dim], tf.float32)
        with tf.variable_scope("s2vt") as scope:
            tf.get_variable_scope().reuse_variables()
            with tf.device('/cpu:0'):
                current_embed = tf.nn.embedding_lookup(self.Wemb, input_feed)
                #current_embed = tf.expand_dims(current_embed, 0)
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1_feed)
            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([output1, current_embed], 1), state2_feed)
            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            words_probabilities = tf.exp(logit_words) / tf.reduce_sum(tf.exp(logit_words), -1)
            max_probs, max_prob_index = tf.nn.top_k(words_probabilities, self.beam_size)
            max_prob_index = tf.cast(tf.reshape(max_prob_index, [-1]), tf.int32)
            max_probs = tf.cast(tf.reshape(max_probs, [-1]), tf.float32)
        word_index = max_prob_index
        probs = max_probs
        return word_index, probs, state2, state1, state2_feed, state1_feed, input_feed

    def build_generator(self, sess, video, length_normalization_factor=0.0):
        video_flat = tf.cast(tf.reshape(video, [-1, self.dim_image]),tf.float32)
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.word_dim])
        state1 = tf.zeros([1, self.lstm1.state_size], tf.float32)
        state2 = tf.zeros([1, self.lstm2.state_size], tf.float32)
        padding = tf.zeros([1, self.word_dim], tf.float32)
        beam_size = self.beam_size
        ########## change encoder ############
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
        ################# encoder #############
        captions = TopN(beam_size*beam_size)
        final_captions = TopN(beam_size)
        initial_word = [1]
        initial_state1 = sess.run(state1)
        initial_state2 = sess.run(state2)

        tf_word_index, tf_probs, tf_state2, tf_state1, state2_feed, state1_feed, input_feed = self.beam_probability()

        word_index, probs, state2, state1 = sess.run([tf_word_index, tf_probs, tf_state2, tf_state1],
                                                     feed_dict={state2_feed: initial_state2, state1_feed: initial_state1, input_feed: initial_word})
        for beam in xrange(beam_size):
            initial_beam = Caption(sentence=[word_index[beam]], img_state = state1, language_state=state2,
                                       logprob=math.log(probs[beam]),
                                       score=math.log(probs[beam]))
            captions.push(initial_beam)
        exclude_num = 0
        for i in range(1, self.n_caption_lstm_step):
            mid_captions = captions.extract(sort=True)[:beam_size]
            captions.reset()
            for mid_caption in mid_captions:
                mid_input_feed = [mid_caption.sentence[-1]]
                mid_language_state_feed = mid_caption.language_state
                mid_img_state_feed = mid_caption.img_state
                word_index, probs,state2, state1 = sess.run([tf_word_index, tf_probs, tf_state2, tf_state1], feed_dict={state2_feed: mid_language_state_feed,state1_feed: mid_img_state_feed, input_feed: mid_input_feed})
                for beam in xrange(beam_size - exclude_num):
                    sentence = mid_caption.sentence + [word_index[beam]]
                    logprob = mid_caption.logprob +math.log(probs[beam])
                    score = logprob
                    if word_index[beam] == 0:
                        if length_normalization_factor > 0:
                            score /= len(sentence) ** length_normalization_factor
                        temp_caption = Caption(sentence, state1, state2, logprob, score)
                        final_captions.push(temp_caption)
                        exclude_num += 1
                    else:
                        temp_caption = Caption(sentence, state1, state2, logprob, score)
                        captions.push(temp_caption)
            if exclude_num == beam_size:
                break
        ###################################
        if not final_captions.size():
            final_captions = captions
        final_cap = final_captions.extract(sort=True)[0]
        tf_sentence = final_cap.sentence
        tf_logprob = final_cap.logprob
        tf_score = final_cap.score
        return tf_sentence, tf_logprob, tf_score


        # return video, tf_sentence

# =====================================================================================
# Global Parameters
# =====================================================================================

# video_train_caption_file = './data/video_corpus.csv'
# video_test_caption_file = './data/video_corpus.csv'

model_path = './models'

video_train_feature_file = '/home/llj/tensorflow_s2vt/5frame_train.txt'

video_test_feature_file = '/home/llj/tensorflow_s2vt/5frame_test.txt'

video_train_sent_file = '/media/llj/storage/all_sentences/msvd_sents_train_lc_nopunc.txt'

video_test_sent_file = '/media/llj/storage/all_sentences/msvd_sents_test_lc_nopunc.txt'

vocabulary_file = '/media/llj/storage/all_sentences/msvd_vocabulary1.txt'
# =======================================================================================
# Train Parameters
# =======================================================================================
dim_image = 1536
lstm_dim = 1000
word_dim = 500

n_lstm_step = 40
n_caption_lstm_step = 35
n_video_lstm_step = 5
beam_size =3

n_epochs = 16
batch_size = 16
start_learning_rate = 0.01
caption_mask_out = open('caption_masks.txt', 'w')


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
            features[video_id].append([float(x) for x in splits[1:]])
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
    caption_mask_out.write('captions: ' + str(caption_batch_ind) + '\n' + 'masks: ' + str(captions_mask) + '\n')
    return caption_batch_ind, captions_mask


def train():  ###### move caption (input_sentence) one column left and also need to move caption_mask (cont_sent)one column left ########################################################llj
    train_captions, train_features = get_video_feature_caption_pair(video_train_sent_file, video_train_feature_file)
    vocabulary = []

    with open(vocabulary_file, 'r') as vocab:
        for line in vocab:
            vocabulary.append(line.rstrip())

    wordtoix, ixtoword = preProBuildWordVocab(vocabulary, word_count_threshold=0)

    if not os.path.exists('./data/wordtoix') or os.path.exists('./data/ixtoword'):
        np.save("./data/wordtoix", wordtoix)
        np.save('./data/ixtoword', ixtoword)

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
    sess = tf.InteractiveSession(config=config)

    # my tensorflow version is 0.12.1, I write the saver with version 1.0
    saver = tf.train.Saver(max_to_keep=100, write_version=1)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                               40000, 0.5, staircase=True)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss, global_step=global_step)
    tf.global_variables_initializer().run()

    # new_saver = tf.train.Saver()
    # new_saver = tf.train.import_meta_graph('./rgb_models/model-1000.meta')
    # new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))

    loss_fd = open('loss.txt', 'w')
    loss_to_draw = []

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

            print 'idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str(
                (time.time() - start_time))
            loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')

        # draw loss curve every epoch
        loss_to_draw.append(np.mean(loss_to_draw_epoch))
        plt_save_dir = "./loss_imgs"
        plt_save_img_name = str(epoch) + '.png'
        plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
        plt.grid(True)
        plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))

        if np.mod(epoch, 2) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

    loss_fd.close()


def test(model_path='/home/llj/tensorflow_s2vt/res_models/best_5frames/batch_size16_5uniform_third_model-11'):
    start_time = time.time()
    test_captions, test_features = get_video_feature_caption_pair(video_test_sent_file, video_test_feature_file)

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
        bias_init_vector=None,beam_size=beam_size)

    # video_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    _ = model.build_model()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.InteractiveSession(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    #init = tf.global_variables_initializer()
    #sess.run(init)
    #for x in tf.get_collection(tf.GraphKeys.VARIABLES):
        #print 'mmm_variable: ',x

    i = 0
    test_output_txt_fd = open('best5frame_beam3.txt', 'w')
    for key, values in test_features.iteritems():
        #if i >249:
            captions_tf, logprob_tf, score_tf = model.build_generator(sess,[test_features[key]])
            generated_word_index = captions_tf
            generated_words = ixtoword[generated_word_index]

            punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
            generated_words = generated_words[:punctuation]

            generated_sentence = ' '.join(generated_words)
            generated_sentence = generated_sentence.replace('<bos> ', '')
            generated_sentence = generated_sentence.replace(' <eos>', '')
            print generated_sentence, '\n'
            test_output_txt_fd.write(key + '\t')
            test_output_txt_fd.write(generated_sentence + '\n')
        #else:
            #i += 1
    end_time = time.time()
    print 'total time: ',(end_time-start_time)

def evaluation(model_path='/home/llj/tensorflow_s2vt/res_models/best_5frames/batch_size16_5uniform_third_model-11'):#home/llj/tensorflow_s2vt/reinforcement_multitask_models/'):
    test_captions, test_features = get_video_feature_caption_pair(video_test_sent_file, video_test_feature_file)

    ixtoword = pd.Series(np.load('./vocab1_data/ixtoword.npy').tolist())
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
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
    greedy_captions, greedy_video = model.build_sampler()

    saver = tf.train.Saver()

    saver.restore(sess, '/home/llj/tensorflow_s2vt/res_models/best_5frames/batch_size16_5uniform_third_model-11')

    with open('baseline_test.txt', 'a') as f:
      #for i in xrange(1000, 25000, 1000):
        #model_path_last = model_path + '5batch_size8reinforce_multitask_model_alpha005-' + str(i)
        #saver.restore(sess, model_path_last)
        all_decoded_for_eval = {}
        test_index = list(range(len(test_captions)))
        random.shuffle(test_index)
        ref_decoded = {}
        for aa in xrange(0,len(set(test_captions[:,0])),batch_size):

            id = list(set(test_captions[:,0]))[aa:aa+batch_size]
            test_video_features = [test_features[x] for x in id]

            feed_dict = {greedy_video: test_video_features}
            greedy_words = sess.run(greedy_captions, feed_dict) #### batch_size x num of each words
            greedy_decoded = decode_captions(np.array(greedy_words), ixtoword)
            for videoid in id:
                if videoid not in all_decoded_for_eval:
                    all_decoded_for_eval[videoid] = []

            [all_decoded_for_eval[x].append(y) for x,y in zip(id,greedy_decoded)]
        #print 'all : ', all_decoded_for_eval

        for key, items in all_decoded_for_eval.iteritems():
            yyy = ' '.join(str(x) for x in items)
            f.write(str(key))
            f.write('\t')
            f.write(yyy)
            f.write('\n')

if __name__ == '__main__':
    args = parse_args()
    if args.task == 'train':
        with tf.device('/gpu:' + str(args.gpu_id)):
            train()
    elif args.task == 'test':
        with tf.device('/cpu:' + str(args.gpu_id)):
            test()
    elif args.task == 'evaluate':
        with tf.device('/gpu:' + str(args.gpu_id)):
            evaluation()

