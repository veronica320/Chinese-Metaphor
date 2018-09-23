from keras.models import load_model
import numpy as np
import copy
import keras.backend as K
from multi_cgru_keras import *
from conf import *
from data_provider import load_data
import time
import operator
from collections import Counter, defaultdict
import os
import tensorflow as tf
from google.protobuf import text_format
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import keras_metrics

# Evaluate model using 10% of training data as testset
def eva_model(train_split_file, nb_class, model_path, model_name, result_path, rel_dic):
	x, x_w, x_jcd, x_med, y, c2vmat, w2vmat = load_data(train_split_file, maxlen, nb_class, rel_dic, mode, input_num, False,
														w2v, ci2v, wi2v, w2vdim, w2idic, c2idic, info, train=True)

	model = load_model(model_path + '/' + model_name + '.model')

	print('Model successfully loaded: {}'.format(model_name))

	time_start = time.time()
	if mode == 'charword':
		res = model.evaluate([x[0], x_w[0]], y, batch_size=batch_size)
	else:
		res = model.evaluate(x, y, batch_size=batch_size)
	time_end = time.time()

	print('Average time cost per sample: ', 1000 * (time_end - time_start) / len(y), 'ms')
	print('Loss and acc: ', res)

	if mode == 'charword':
		pred = model.predict([x[0], x_w[0]])
	else:
		pred = model.predict(x)

	y_true = np.argmax(y, axis=1)
	y_pred = np.argmax(pred, axis=1)
	f_score = round(f1_score(y_true=y_true, y_pred=y_pred, average='macro'), 4)
	print('fscore:', f_score)

	n = 0
	num_top1, num_top3, num_topN = 0, 0, 0
	err_top1, err_top3, err_topN = [], [], []
	real_lab = defaultdict(int)
	err_mat = defaultdict(int)

	rel_dic_rev = dict(zip(rel_dic.values(), rel_dic.keys()))

	if not os.path.isdir(result_path):
		os.mkdir(result_path)

	with open(train_split_file, 'r') as inf, open(result_path + '/' + model_name + '_score.txt', 'w') as ouf1,\
			open(result_path + '/' + model_name + '_matrix.txt', 'w') as ouf2:
		ouf1.write('\t'.join(['Sentence', 'True_label', 'Pred_label', 'Score', 'Correct']) + '\n')
		for line in inf:
			line = line.strip('\n').split('\t')[1:]
			q, l, pr = line[0], rel_dic[line[1]], pred[n]
			n += 1
			real_lab[line[1]] += 1

			if l != np.argmax(pr):
				cor = 0
			else:
				cor = 1
			pr_dic = dict(zip([rel_dic_rev[i] for i in range(len(pr))], pr))
			pr_top = sorted(pr_dic.items(), key=operator.itemgetter(1), reverse=True)[0]
			ouf1.write('\t'.join([q, line[1], pr_top[0], str(pr_top[1]), str(cor)]) + '\n')
			if l != np.argmax(pr):
				err_top1.append(line[1])
				err_mat[(line[1], str(pr_top[0]))] += 1

		err_mat = sorted(err_mat.items(), key=operator.itemgetter(1), reverse=True)
		ouf2.write('\t'.join(['True_label', 'Pred_label', 'Err_count', 'Err_ratio']) + '\n')
		for pair, count in err_mat:
			ouf2.write('\t'.join([pair[0], pair[1], str(count), str(count / Counter(err_top1)[pair[0]])]) + '\n')

		err_top1 = sorted(Counter(err_top1).items(), key=operator.itemgetter(1), reverse=True)
		ouf2.write('\n' + '\t'.join(['True_label', 'Err_count', 'All_count', 'Accuracy']) + '\n')
		for rela, count in err_top1:
			ouf2.write('\t'.join([rela, str(count), str(real_lab[rela]), str(1 - count/real_lab[rela])]) + '\n')

		ouf2.write('\n' + 'Overall Accuracy:' + '\t' + str(res[1]))

#######################################################################################################################

eva_model(trainsplitfile, nb_class, model_path, model_name, result_path, rel_dic)
