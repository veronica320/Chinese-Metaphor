import json
import sys

# task1:verb or task2:emo
key = ['verb', 'emo'][1]

# whether to use data extended with back translation or not
extended = ['', '_extended'][1]

# use 90% tranining data to train
trainfile = '../data/tokenized_data/{}_train_0.9{}_tokenized.txt'.format(key, extended)
trainsplitfile = '../data/tokenized_data/{}_train_0.1_tokenized.txt'.format(key)
valfile = None
testfile = '../data/tokenized_data/{}_test_tokenized.txt'.format(key)

# path to lexicon
lexicon = '../lexicon/lexicon.txt'

#  number of filters for CNN kernels
nb_filter = [128, 256, 128]

#  CNN kernel size
wsize = [1, 2, 3]

#  max length for input questions
maxlen = 40

# number of units for dense layers
denseuni = 512

# drop out ratio for CNN layers
cnn_dropout = 0.1

#  l2 regularization ratio
l2 = 0.01

# learning rate
lr = 0.001

#  whether to use pre-trained w2v
w2v = [True, False][0]

#  dimension of pre-trained w2v
w2vdim = 300

# which pre-trained word embedding to use
w2vname = ['wiki', 'baidu', 'lit'][0]

# character-based or word-based
mode = ['char', 'word', 'charword'][2]

# True if input is already segmented
# segged = [True, False][1]

# Whether to use extra info line Jaccard similarity
info = [True, False][1]

# Whether input sentence is 1 or 2
input_num = [1, 2][0]

ci2v = wi2v = c2idic = w2idic = cvsize = wvsize = None
if mode == 'char':
    ci2v = None
    with open('../dicts/vocab_char.json', 'r') as f:
        c2idic = json.load(f)
    cvsize = len(c2idic)
    if w2v:
        with open('../pretrained_emb/' + w2vname + '_char.json') as f:
            ci2v = json.load(f)
elif mode == 'word':
    wi2v = None
    with open('../dicts/vocab_word.json', 'r') as f:
        w2idic = json.load(f)
    wvsize = len(w2idic)
    if w2v:
        with open('../pretrained_emb/' + w2vname + '_word.json', 'r') as f:
            wi2v = json.load(f)
elif mode == 'charword':
    ci2v = None
    with open('../dicts/vocab_char.json', 'r') as f:
        c2idic = json.load(f)
    cvsize = len(c2idic)
    if w2v:
        with open('../pretrained_emb/' + w2vname + '_char.json') as f:
            ci2v = json.load(f)
    wi2v = None
    with open('../dicts/vocab_word.json', 'r') as f:
        w2idic = json.load(f)
    wvsize = len(w2idic)
    if w2v:
        with open('../pretrained_emb/' + w2vname + '_word.json', 'r') as f:
            wi2v = json.load(f)

# property json file; format: {property: index}
with open('../dicts/{}_relation.json'.format(key), 'r') as rf:
    rel_dic = json.load(rf)
nb_class = len(rel_dic)

# gpu device
gpu = 0

# train/dev split ratio when training
split = 0.1

# shuffle training data
shuffle = True

# batch size
batch_size = 200

#  epoch
epoch = 30

#  patience for early-stopping
patience = 5

#  monitor for early-stopping and checkpoint
monitor = 'val_loss'

#  smooth factor for category weight; see function get_class_weights in train.py
smooth_factor = 0.001

# model name
model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(key, mode, 'w2v' if w2v else 'rand', split, w2vname, smooth_factor, l2, cnn_dropout, extended)

# model path
model_path = '../models/' + model_name.split('_')[0] + '/' + '_'.join(model_name.split('_')[1:])

# result path
result_path = '../results/' + model_name.split('_')[0] + '/' + '_'.join(model_name.split('_')[1:])
