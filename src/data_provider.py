import re
from keras.utils import np_utils
from collections import Counter
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import operator
import copy

# tokenize a sentence with space
def tokenize(string, mode='word'):
	if mode == 'char':
		return re.sub('\s+', ' ', ' '.join(list(string)))  # 'a b c .'
	else:
		return string

# get sentence list and label list from data file
def read_csv(datafile, mode, input_num, rel_dic, train=True):
	lines = list(open(datafile, "r").readlines())

	sent_lists = {}
	for i in range(input_num):
		sent_lists['{}'.format(i)] = []

	lab_list = []
	for line in lines:
		try:
			line = line.strip('\n').split('\t')[1:]
			if line:
				for i in range(input_num):
					sent = tokenize(line[i], mode)
					sent_lists['{}'.format(i)].append(sent)
				if train:
					lab = line[-1]
					try:
						lab = rel_dic[lab]
					except:
						lab = int(lab)
					lab_list.append(lab)
		except Exception as e:
			print(e)
			print(line)
	return sent_lists, lab_list

# transform a sentence to a list of word indices, and do padding if sent length is smaller than maxlen
def w2i(txt, maxlen, w2idic):  # txt = ['a b', 'c d'], w2idic = {'a':1}
	for e, i in enumerate(txt):  # i = 'a b'
		txt[e] = i.split(' ')  # txt[e] = ['a', 'b']
		for p, k in enumerate(txt[e]):  # k = 'a'
			txt[e][p] = w2idic[k] if k in w2idic else 0
	return pad_sequences(txt, maxlen=maxlen, padding='post', truncating='post')

# map char and word to index
def cw2i(txt, maxlen, c2idic, w2idic):  
	x_k = []
	x_w_k = []
	for e, i in enumerate(txt):  # 遍历每一句
		#处理到字index的映射
		cur_list = ([x for x in i.replace(' ','')])
		for p, k in enumerate(cur_list):
			cur_list[p] = c2idic[k] if k in c2idic else 0
		x_k.append(cur_list)
		#处理到词index的映射
		cur_list = i.split(" ")
		index_list = []
		for p, k in enumerate(cur_list):  # 遍历每一个分词
			for item in k:
				index_list.append(w2idic[k] if k in w2idic else 0)
		x_w_k.append(index_list)
	return pad_sequences(x_k, maxlen=maxlen, padding='post', truncating='post'), pad_sequences(x_w_k, maxlen=maxlen, padding='post', truncating='post')

# build word embedding matrix
def build_w2v(i2v, w2idic, dim):
	vsize = len(w2idic)
	embedding_matrix = np.random.uniform(-0.5, 0.5, (vsize+1, dim))
	for word, i in w2idic.items():
		embedding_vector = i2v.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be randomly initiated
			try:
				embedding_matrix[i] = embedding_vector
			except:
				pass
	return embedding_matrix

def load_data(datafile, maxlen, nb_class, rel_dic, mode, input_num, shuffle, w2v, ci2v, wi2v, w2vdim, w2idic, c2idic, info=False, train=True):
	print('Loading training data...')
	q, label = read_csv(datafile, mode, input_num, rel_dic, train)  # q = {'1': ['a b', 'x y'], '2': [q3, q4]}
	nb_class_train = len(set(label))
	nb_train = [(i, Counter(label)[i]) if i in Counter(label) else (i, 0) for i in range(nb_class)]
	nb_train = sorted(nb_train, key=operator.itemgetter(1), reverse=True)

	if shuffle:
		label_shuffle = []
		indices = list(range(len(q['0'])))
		np.random.shuffle(indices)
		for i in indices:
			label_shuffle.append(label[i])
		label = label_shuffle
		for key in q:  # key = '1'
			q_shuffle = []
			for i in indices:
				q_shuffle.append(q[key][i])
			q[key] = q_shuffle
		print('Data shuffled!')

	y = np_utils.to_categorical(np.asarray(label), nb_class)
	x = [] # 单独字或单独词的index
	x_w = [] # 字词拼接时词部分的index，比如q='ab c d', c2i=[a:1,b:2,c:3,d:4], w2i=[ab:1,c:2,d:3], 则x_w=[1,1,2,3]
	x_jcd = []
	x_med = []
	for k in q:
		if mode == "char":
			x_k = w2i(q[k], maxlen, c2idic)
			x.append(x_k)
		if mode == "word":
			x_k = w2i(q[k], maxlen, w2idic)
			x.append(x_k)
		if mode == "char_word":
			x_k, x_w_k = cw2i(q[k], maxlen, c2idic, w2idic)
			x.append(x_k)
			x_w.append(x_w_k)

	print('Training data successfully loaded!')
	print('Number of classes in the training data: ' + str(nb_class_train))
	print('Number of samples in each class: ' + str(nb_train))

	if not w2v:
		return x, x_w, x_jcd, x_med, y, 0, 0  # label = arr[[[0,1], [1,0]]], txt = arr[[[1,2], [3,4]]]
	else:
		print('Using pre-trained w2v, calculating weights...')
		if mode == 'char':
			char_mat = build_w2v(ci2v, c2idic, w2vdim)
			word_mat = 0
		if mode == 'word':
			char_mat = 0
			word_mat = build_w2v(wi2v, w2idic, w2vdim)
		if mode == 'char_word':
			char_mat = build_w2v(ci2v, c2idic, w2vdim)
			word_mat = build_w2v(wi2v, w2idic, w2vdim)
		print('w2v weights done!')
		return x, x_w, x_jcd, x_med, y, char_mat, word_mat

