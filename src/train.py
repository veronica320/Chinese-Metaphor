import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from keras.utils import plot_model, np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
import multi_cgru_keras
from collections import Counter
from conf import *
from data_provider import load_data
import numpy as np
import os


def get_class_weights(y, smooth_factor=0.0):
	"""
	Returns the weights for each class based on the frequencies of the samples
	:param smooth_factor: factor that smooths extremely uneven weights
	:param y: list of true labels (the labels must be hashable)
	:return: dictionary with the weight for each class
	"""
	if smooth_factor > 9999:
		return 'None'
	else:
		y = [np.argmax(i) for i in y]
		counter = Counter(y)

		if smooth_factor > 0:
			p = max(counter.values()) * smooth_factor
			for k in counter.keys():
				counter[k] += p

		majority = max(counter.values())
		weight = {cls: float(majority / count) for cls, count in counter.items()}

		return weight



def train_model(gpu, input_num, nb_class, rel_dic, trainfile, valfile, model_path, model_name, epoch):

	x, x_w, x_jcd, x_med, y, c2vmat, w2vmat = load_data(trainfile, maxlen, nb_class, rel_dic, mode, input_num, shuffle,
														w2v, ci2v, wi2v, w2vdim, w2idic, c2idic, info, train=True)
	if valfile:
		x_v, x_w_v, x_jcd_v, x_med_v, y_v, c2vmat_v, w2vmat_v = load_data(valfile, maxlen, nb_class, rel_dic, mode,
																		  input_num, shuffle, w2v, ci2v, wi2v, w2vdim,
																		  w2idic, c2idic, info, train=True)
		val = [x_v, y_v]


	else:
		val = None

	print('Building base model: {}'.format(model_name))

	model = multi_cgru_keras.creat_model(gpu, model_name, mode, nb_class, nb_filter, wsize, maxlen, cvsize, wvsize,
										 denseuni, cnn_dropout, l2, lr, w2v, w2vdim, c2vmat, w2vmat)
	plot_model(model, show_shapes=True, to_file='model_cgru_cw.png')

	if not os.path.isdir(model_path):
		os.mkdir(model_path)

	checkpoint = ModelCheckpoint(model_path + '/' + model_name + '.model', monitor=monitor, verbose=1,
								 save_weights_only=False, save_best_only=True, mode='auto')

	earlystop = EarlyStopping(monitor=monitor, patience=patience, verbose=1, mode='auto')

	print('Begin training model: {}'.format(model_name))


	if mode == 'charword':
		history = model.fit([x[0], x_w[0]], y,
							validation_data=val,
							batch_size=batch_size,
							epochs=epoch,
							callbacks=[checkpoint, earlystop],
							validation_split=split,
							class_weight=get_class_weights(y, smooth_factor))
	else:
		print(x[0][0])

		history = model.fit(x, y,
							validation_data=val,
							batch_size=batch_size,
							epochs=epoch,
							callbacks=[checkpoint, earlystop],
							validation_split=split,
							class_weight=get_class_weights(y, smooth_factor))

	print('Model successfully trained: {}'.format(model_name))

	return model, history


def plot_training(history, name):
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = [i + 1 for i in range(len(acc))]
	with PdfPages(name + '.pdf') as pdf:
		plt.figure()
		plt.subplot(211)
		plt.plot(epochs, loss, 'r-', label='train_loss')
		plt.plot(epochs, val_loss, 'b-', label='val_loss')
		plt.ylabel('loss')
		plt.legend(loc='upper right')

		plt.subplot(212)
		plt.plot(epochs, acc, 'r:', label='train_acc')
		plt.plot(epochs, val_acc, 'b:', label='val_acc')
		plt.xlabel('epoch')
		plt.ylabel('acc')
		plt.legend(loc='upper left')

		pdf.savefig()
		plt.close()


#######################################################################################################################

model, history = train_model(gpu, input_num, nb_class, rel_dic, trainfile, valfile, model_path, model_name, epoch)

plot_training(history, model_path + '/' + model_name)
