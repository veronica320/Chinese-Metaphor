from conf import *
from data_provider import load_data
import numpy as np
from keras.models import load_model
from multi_cgru_keras import *

# Write predictions for test data to csv file
def generate_test_labels(testfile):
	x, x_w, x_jcd, x_med, y, c2vmat, w2vmat = load_data(testfile, maxlen, nb_class, rel_dic, mode, input_num,
	                                                    False, w2v, ci2v, wi2v, w2vdim, w2idic, c2idic, info, train=False)
	model = load_model(model_path + '/' + model_name + '.model', custom_objects={'fscore': fscore})

	y_pred = np.argmax(model.predict(x), axis=1)

	inf = open(testfile, 'r')
	ouf = open(result_path + '/' + model_name + '_pred.csv', 'w')
	ouf.write('ID\tLabel\n')

	i = 0
	for line in inf:
		ouf.write(line.split()[0] + '\t')
		ouf.write(str(y_pred[i])+'\n')
		i += 1

#######################################################################################################################

generate_test_labels(testfile)