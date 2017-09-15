import numpy as np
import scipy.io
import sys
import argparse
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Reshape
from keras.layers import Merge
from keras.layers.recurrent import LSTM

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import model_from_json

from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, RemoteMonitor

from sklearn.externals import joblib
from sklearn import preprocessing

from spacy.en import English
from progressbar import Bar, ETA, Percentage, ProgressBar
import matplotlib.pyplot as plt

from utils import grouper, selectFrequentAnswers
from features import *
from Validation import *

def main():
	print 'Train LSTM encoder + MLP decoder'
	parser = argparse.ArgumentParser()
	parser.add_argument('-num_hidden_units_mlp', type=int, default=1024)
	parser.add_argument('-num_hidden_units_lstm', type=int, default=512)
	parser.add_argument('-num_hidden_layers_mlp', type=int, default=3)
	parser.add_argument('-num_hidden_layers_lstm', type=int, default=1)
	parser.add_argument('-dropout', type=float, default=0.5)
	parser.add_argument('-activation_mlp', type=str, default='tanh')
	parser.add_argument('-num_epochs', type=int, default=100)
	parser.add_argument('-model_save_interval', type=int, default=5)
	parser.add_argument('-batch_size', type=int, default=4096)
	parser.add_argument('-gap_layer_units', type = int, default = 1024)
	#TODO Feature parser.add_argument('-resume_training', type=str)
	#TODO Feature parser.add_argument('-language_only', type=bool, default= False)
	args = parser.parse_args()

	word_vec_dim= 300
	img_dim = 4096
	max_len = 30
	nb_classes = 1000

	#get the data
	questions_train = open('../data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
	questions_lengths_train = open('../data/preprocessed/questions_lengths_train2014.txt', 'r').read().decode('utf8').splitlines()
	answers_train = open('../data/preprocessed/answers_train2014_modal.txt', 'r').read().decode('utf8').splitlines()
	images_train = open('../data/preprocessed/images_train2014.txt', 'r').read().decode('utf8').splitlines()
	answers_train_all = open('../data/preprocessed/answers_train2014_all.txt', 'r').read().decode('utf8').splitlines()
	vgg_model_path = '../features/coco/vgg_feats.mat'

	max_answers = nb_classes
	questions_train, answers_train, answers_train_all, images_train = selectFrequentAnswers(questions_train,answers_train, answers_train_all, images_train, max_answers)
	questions_lengths_train, questions_train, answers_train, answers_train_all, images_train = (list(t) for t in zip(*sorted(zip(questions_lengths_train, questions_train, answers_train, answers_train_all, images_train))))

	questions_val = open('../data/preprocessed/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
	questions_lengths_val = open('../data/preprocessed/questions_lengths_val2014.txt', 'r').read().decode('utf8').splitlines()
	answers_val = open('../data/preprocessed/answers_val2014_all.txt', 'r').read().decode('utf8').splitlines()
	images_val = open('../data/preprocessed/images_val2014_all.txt', 'r').read().decode('utf8').splitlines()
	vgg_model_path = '../features/coco/vgg_feats.mat'
	
	questions_lengths_val, questions_val, answers_val, images_val = (list(t) for t in zip(*sorted(zip(questions_lengths_val, questions_val, answers_val, images_val))))

	#encode the remaining answers
	labelencoder = preprocessing.LabelEncoder()
	labelencoder.fit(answers_train)
	nb_classes = len(list(labelencoder.classes_))
	joblib.dump(labelencoder,'../models/labelencoder.pkl')
	
	image_input = Input(shape = (img_dim, ), name = 'image_input')
	# image_gap = Dense(args.gap_layer_units, activation = args.activation_mlp)(image_input)

	language_input = Input(shape = (None, word_vec_dim), name = 'language_input')	
	lstm_out = LSTM(args.num_hidden_units_lstm)(language_input)
	# lstm_gap = Dense(args.gap_layer_units, activation = args.activation_mlp)(lstm_out)
	
	# x = keras.layers.concatenate([lstm_gap, image_gap])
	x = keras.layers.concatenate([lstm_out, image_input])
	x = Dense(1024, activation = args.activation_mlp)(x)
	x = Dropout(args.dropout)(x)
	x = Dense(512, activation = args.activation_mlp)(x)
	x = Dropout(args.dropout)(x)
	x = Dense(256, activation = args.activation_mlp)(x)
	x = Dropout(args.dropout)(x)
	main_output = Dense(nb_classes, activation = 'softmax', name = 'main_output')(x)
	model = Model(inputs = [language_input, image_input], outputs = [main_output])

	# args.model = '../models/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_num_hidden_layers_lstm_1.json'
	# args.weights = '../models/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_num_hidden_layers_lstm_1_epoch_100.hdf5'

	# model = model_from_json(open(args.model).read())
	# model.load_weights(args.weights)

	json_string = model.to_json()
	model_file_name = '../models/lstm_1_num_hidden_units_lstm_' + str(args.num_hidden_units_lstm) + \
						'_num_hidden_units_mlp_' + str(args.num_hidden_units_mlp) + '_num_hidden_layers_mlp_' + \
						str(args.num_hidden_layers_mlp) + '_num_hidden_layers_lstm_' + str(args.num_hidden_layers_lstm)
	results_path = '../results/lstm_decoder_1_num_hidden_units_lstm_' + str(args.num_hidden_units_lstm) + \
						'_num_hidden_units_mlp_' + str(args.num_hidden_units_mlp) + '_num_hidden_layers_mlp_' + \
						str(args.num_hidden_layers_mlp) + '_num_hidden_layers_lstm_' + str(args.num_hidden_layers_lstm)
	open(model_file_name + '.json', 'w').write(json_string)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	print 'Compilation done'

	features_struct = scipy.io.loadmat(vgg_model_path)
	VGGfeatures = features_struct['feats']
	print 'loaded vgg features'
	image_ids = open('../features/coco_vgg_IDMap.txt').read().splitlines()
	img_map = {}
	for ids in image_ids:
		id_split = ids.split()
		img_map[id_split[0]] = int(id_split[1])

	nlp = English()
	print 'loaded word2vec features...'
	## training
	print 'Training started...'

	Acc_train = [0] * args.num_epochs
	Acc_val = [0] * args.num_epochs
	loss_list = [0] * args.num_epochs

	f1 = open('../results/loss_accuracy_lstm_encoder.txt', 'a')
	f1.write(model_file_name + '\n')
	for k in xrange(args.num_epochs):

		print str(400 + k + 1) + 'th Iteration ...'

		progbar = generic_utils.Progbar(len(questions_train))
		loss_sum = 0
		it = 0
		for qu_batch,an_batch,im_batch in zip(grouper(questions_train, args.batch_size, fillvalue=questions_train[-1]), 
												grouper(answers_train, args.batch_size, fillvalue=answers_train[-1]), 
												grouper(images_train, args.batch_size, fillvalue=images_train[-1])):
			timesteps = len(nlp(qu_batch[-1])) #questions sorted in descending order of length
			X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, timesteps)
			X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
			X_i_batch_normalized = preprocessing.normalize(X_i_batch, norm='l2')
			#print X_i_batch.shape, X_q_batch.shape
			Y_batch = get_answers_matrix(an_batch, labelencoder)
			loss = model.train_on_batch([X_q_batch, X_i_batch_normalized], Y_batch)
			progbar.add(args.batch_size, values=[("train loss", loss)])
			it += 1
			loss_sum += loss
		print " " + str(loss_sum / float(it))

		if (k + 1)%args.model_save_interval == 0:
			model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k + 1))

		loss_list[k] = loss_sum /float(it)
		f1.write(str(loss_list[k]) + ' ')

		# print '    Results on Training set: '
		# Acc_train[k] = Validation_LSTM_encoder(model, questions_train, answers_train_all, images_train, img_map ,VGGfeatures, labelencoder, \
		# 	args.batch_size, nlp, nb_classes, results_path+'_train', model_file_name+'_train')
		print '    Results on Validation set: '
		Acc_val[k] = Validation_LSTM_encoder(model, questions_val, answers_val, images_val, img_map ,VGGfeatures, labelencoder, \
			args.batch_size, nlp, nb_classes, results_path, model_file_name)
		f1.write(str(Acc_val[k]) + '\n')
	
	f1.close()

	plt.figure(1)
	plt.xlabel('Iterations')
	plt.ylabel('Accuracy')
	plt.title('Accuracy on Training and Validation set')
	# plt.plot(range(args.num_epochs), Acc_train, 'b-', label = 'Accuracy on Training set')
	# plt.hold(True)
	plt.plot(range(args.num_epochs), Acc_val, 'r--', label = 'Accuracy on Validation set')
	plt.legend(loc = 'lower right')
	plt.savefig('accuracy_train_val.png')

	plt.figure(2)
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.title('Convergence curve')
	plt.plot(range(args.num_epochs), loss_list, 'r--')
	plt.savefig('Convergence_curve.png')

	model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k+1))
	
if __name__ == "__main__":
	main()