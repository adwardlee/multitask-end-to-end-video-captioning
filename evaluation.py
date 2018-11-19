import cPickle as pickle
import os
import sys
sys.path.append('/data1/lijun/caption-eval/coco-caption')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor



def score_all(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores


def score(ref, hypo):
    scorers = [
        (Meteor(), "METEOR")
    ]
    #scorers = [
    #    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
    #]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores


def evaluate_for_particular_captions(cand, ref_captions):
    ref = ref_captions
    # with open(candidate_path, 'rb') as f:
    #     cand = pickle.load(f)

    # make dictionary
    hypo = {}
    refe = {}
    for key, caption in cand.iteritems():
        hypo[key] = cand[key]
        refe[key] = ref[key]
    # compute bleu score
    final_scores = score_all(refe, hypo)

    # print out scores

    return final_scores

def evaluate_captions_wrapper(args):
   return evaluate_captions(*args)

def evaluate_captions(ref, cand):
    ref = [ref]
    cand = [cand]
    hypo = {}

    refe = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
        refe[i] = ref[i]
    final_scores = score(refe, hypo)
    # return final_scores['Bleu_1']
    #return 1 * final_scores['CIDEr']
    return final_scores['METEOR']


def evaluate(data_path='./data', split='val', get_scores=False):
    reference_path = os.path.join(data_path, "%s/%s.references.pkl" % (split, split))
    candidate_path = os.path.join(data_path, "%s/%s.candidate.captions.pkl" % (split, split))

    # load caption data
    with open(reference_path, 'rb') as f:
        ref = pickle.load(f)
    with open(candidate_path, 'rb') as f:
        cand = pickle.load(f)

    # make dictionary
    hypo = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]

    # compute bleu score
    final_scores = score_all(ref, hypo)

    # print out scores
    print 'Bleu_1:\t', final_scores['Bleu_1']
    print 'Bleu_2:\t', final_scores['Bleu_2']
    print 'Bleu_3:\t', final_scores['Bleu_3']
    print 'Bleu_4:\t', final_scores['Bleu_4']
    print 'METEOR:\t', final_scores['METEOR']
    print 'ROUGE_L:', final_scores['ROUGE_L']
    print 'CIDEr:\t', final_scores['CIDEr']

    if get_scores:
        return final_scores

def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<eos>':
                #words.append('.')
                break
            else:
                words.append(word)
        decoded.append(' '.join(words))
    return decoded

def decode_captions_masks(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    masks = []
    for i in range(N):
        words = []
        mask = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<eos>':
                #words.append('.')
                mask.append(1)
                break
            else:
                words.append(word)
                mask.append(1)
        decoded.append(' '.join(words))
        mask.extend([0]*(T-len(mask)))
        masks.append(mask)
    return masks, decoded
