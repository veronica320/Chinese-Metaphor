import json
import re

# read sentences from file
def read_sents(file):
	sents = []
	for line in file:
		sents.append(line.split('\t')[1])
	return sents

# get vocabulary from a list of sentences
def get_vocab(sents):
	vocab = set()
	for sent in sents:
		for v in sent:
			vocab.add(v)
	return vocab

# filter pretrained word embeddings with vocabulary, and create a json version
def filter_pretrained_emb(vocab, w2v_file_name):
	w2v_original = open(w2v_file_name, 'r')
	w2v_filtered = open(w2v_file_name + "_filter", 'w')
	w2v_dict = {}
	w2v_json = open(w2v_file_name + "_json", 'w')
	for line in w2v_original:
		cols = line.split()
		if cols[0] in vocab:
			w2v_filtered.write(line)
			w2v_dict[cols[0]] = [float(value) for value in cols[1:]]
	json.dump(w2v_dict, w2v_json)


# data_path = '../data/'
# file_names = ['verb_train_all.txt', 'verb_test.txt', 'emo_train_extended.txt', 'emo_test.txt']
#
# sents =[]
# for name in file_names:
# 	sents += read_sents(open(data_path+name, 'r'))
#
# vocab = get_vocab(sents)
#
# print('Vocabulary size: ', len(vocab))
#
# w2v_file_name = '../pretrained_emb/sgns.wiki.char'
# filter_pretrained_emb(vocab, w2v_file_name)
