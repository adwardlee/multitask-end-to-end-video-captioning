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
#from evaluation import *
from cider_evaluation import *
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


# =====================================================================================
# Global Parameters
# =====================================================================================

# video_train_caption_file = './data/video_corpus.csv'
# video_test_caption_file = './data/video_corpus.csv'

model_path = './res_models'

video_train_feature_file = '/home/llj/tensorflow_s2vt/tf_inceptionres_v2_train_noval_feature.txt'

video_test_feature_file = '/home/llj/tensorflow_s2vt/tf_inceptionres_v2_test_feature.txt'

#video_train_feature_file = '/media/llj/storage/all_sentences/msvd_inception_globalpool_train_origin.txt'

#video_test_feature_file = '/media/llj/storage/all_sentences/msvd_inception_globalpool_test_origin.txt'

video_train_sent_file = '/media/llj/storage/all_sentences/msvd_sents_test_lc_nopunc.txt'

video_test_sent_file = '/media/llj/storage/all_sentences/msvd_sents_test_lc_nopunc.txt'

#vocabulary_file = '/media/llj/storage/all_sentences/coco_msvd_allvocab.txt'
vocabulary_file = '/media/llj/storage/all_sentences/msvd_vocabulary1.txt'
# =======================================================================================
# Train Parameters
# =======================================================================================
dim_image = 1536
#dim_image = 1024
lstm_dim = 1000
word_dim = 500

n_lstm_step = 60
n_caption_lstm_step = 35
n_video_lstm_step = 25

n_epochs = 15
batch_size = 16
start_learning_rate = 0.01
#caption_mask_out = open('caption_masks.txt', 'w')

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def get_video_feature_caption_pair(sent_file=video_train_sent_file):
    sents = {}
    features = {}
    vid = []
    with open(sent_file, 'r') as video_sent_file:
        for line in video_sent_file:
            line = line.strip()
            id_sent = line.split('\t')
            if id_sent[0] not in sents:
              sents[id_sent[0]] = []
              vid.append(id_sent[0])
            sents[id_sent[0]].append(id_sent[1])
    return sents, vid


def train():  ###### move caption (input_sentence) one column left and also need to move caption_mask (cont_sent)one column left ########################################################llj
    train_captions, train_vid = get_video_feature_caption_pair(video_train_sent_file)
    vocabulary = []
    test_captions, test_vid = get_video_feature_caption_pair(video_test_sent_file)
    last_cider_score = 0

    with open(vocabulary_file, 'r') as vocab:
        for line in vocab:
            vocabulary.append(line.rstrip())

    ############# test before training########

    #greedy_captions, greedy_video_features = model.build_sampler()
    ######
    best_captions = {}
    for video_id in train_vid:
        cider_score = 0
        best_captions[video_id] = []
        b = {}
        b[0]=train_captions[video_id]
        for one_captions in train_captions[video_id]:
          a = []      
          a.append(one_captions)
          scores = evaluate_captions_cider(b, a)
          print 'cider: ', scores
          if scores > cider_score:
            one_best_captions = one_captions
            cider_score = scores
        best_captions[video_id].append(one_best_captions)
        print '{},{},{}'.format(video_id,one_best_captions,cider_score)
        last_cider_score += cider_score
    print 'last cider: ',last_cider_score/len(train_vid)

    #with open('msvd_best_captions','a') as f:
      #for key in natural_sort(best_captions.iterkeys()):
        #f.write(key)
        #f.write('\t')
        #f.write(' '.join(best_captions[key]))
        #f.write('\n')

def train1():  ###### move caption (input_sentence) one column left and also need to move caption_mask (cont_sent)one column left ########################################################llj
    train_captions, train_vid = get_video_feature_caption_pair(video_train_sent_file)
    vocabulary = []
    #test_captions, test_features, test_vid = get_video_feature_caption_pair(video_test_sent_file, video_test_feature_file)

    with open(vocabulary_file, 'r') as vocab:
        for line in vocab:
            vocabulary.append(line.rstrip())

    ############# test before training########

    #greedy_captions, greedy_video_features = model.build_sampler()
    ######
    best_captions = {}
    for video_id in train_vid:
        start = time.time()
        cider_score = 0.0001
        best_captions[video_id] = []
        b = {}
        b[0]=train_captions[video_id]
        one_best_captions = ''
        for x in xrange(25):
          sub_cider_score = 0
          one_word_start_time = time.time()

          for word in vocabulary:
            temp_caption = one_best_captions
            temp_caption = temp_caption + word + ' '
            scores = evaluate_captions_cider(b, [temp_caption])
            if scores > sub_cider_score:
                best_word = word
                sub_cider_score = scores

          if sub_cider_score <= cider_score:
              break
          cider_score = sub_cider_score
          one_best_captions = one_best_captions + best_word + ' '
          print 'best word: ', best_word
          print 'one word time: ',(time.time() - one_word_start_time)
          print 'cider_score: ',cider_score
        best_captions[video_id].append(one_best_captions)
        print '{},{},{}'.format(video_id, one_best_captions, cider_score)
        print 'time: ',(time.time()-start)

    with open('msvd_best_captions','a') as f:
      for key in natural_sort(best_captions.iterkeys()):
        f.write(key)
        f.write('\t')
        f.write(' '.join(best_captions[key]))
        f.write('\n')


if __name__ == '__main__':
    args = parse_args()
    if args.task == 'train':
            train()
    if args.task == 'train1':
            train1()

