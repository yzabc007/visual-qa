import sys
from random import shuffle
import argparse

import numpy as np
import scipy.io

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras import optimizers

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import model_from_json

from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from spacy.en import English
import matplotlib.pyplot as plt

from features import *
from utils import grouper, selectFrequentAnswers
from Validation import *

def main():
	print 'Train MLP'
	parser = argparse.ArgumentParser()
	parser.add_argument('-featureType', type=str, default='BoW') #BoW, WordsGlove, SentGlove
	parser.add_argument('-num_hidden_units', type=int, default=1024)
	parser.add_argument('-num_hidden_layers', type=int, default=3)
	parser.add_argument('-dropout', type=float, default=0.5)
	parser.add_argument('-activation', type=str, default='tanh')
	parser.add_argument('-language_only', type=bool, default= False)
	parser.add_argument('-num_epochs', type=int, default=2000)
	parser.add_argument('-model_save_interval', type=int, default=10)
	parser.add_argument('-batch_size', type=int, default=2048)
	parser.add_argument('-num_top_all_words', type = int, default = 1000)
	parser.add_argument('-num_top_start_words', type = int, default = 10)
	parser.add_argument('-num_start_words', type = int, default = 3)
	args = parser.parse_args()

	questions_train = open('../data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
	answers_train = open('../data/preprocessed/answers_train2014_modal.txt', 'r').read().decode('utf8').splitlines()
	answers_train_all = open('../data/preprocessed/answers_train2014_all.txt', 'r').read().decode('utf8').splitlines()
	images_train = open('../data/preprocessed/images_train2014.txt', 'r').read().decode('utf8').splitlines()
	vgg_model_path = '../features/coco/vgg_feats.mat'

	maxAnswers = 1000
	questions_train, answers_train, answers_train_all, images_train = selectFrequentAnswers(questions_train,answers_train, answers_train_all, images_train, maxAnswers)
	
	# print [answers_train.count(answers_train[i]) for i in range(1000)]
	print max([answers_train.count(answers_train[i]) for i in range(1000)])
	print min([answers_train.count(answers_train[i]) for i in range(1000)])
	print np.mean([answers_train.count(answers_train[i]) for i in range(1000)])

	questions_val = open('../data/preprocessed/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
	questions_lengths_val = open('../data/preprocessed/questions_lengths_val2014.txt', 'r').read().decode('utf8').splitlines()
	answers_val = open('../data/preprocessed/answers_val2014_all.txt', 'r').read().decode('utf8').splitlines()
	images_val = open('../data/preprocessed/images_val2014_all.txt', 'r').read().decode('utf8').splitlines()
	vgg_model_path = '../features/coco/vgg_feats.mat'

	#encode the remaining answers
	labelencoder = preprocessing.LabelEncoder()
	labelencoder.fit(answers_train)
	nb_classes = len(list(labelencoder.classes_))
	joblib.dump(labelencoder,'../models/labelencoder.pkl')

	features_struct = scipy.io.loadmat(vgg_model_path)
	VGGfeatures = features_struct['feats']
	print 'loaded vgg features'
	image_ids = open('../features/coco_vgg_IDMap.txt').read().splitlines()
	img_map = {}
	for ids in image_ids:
		id_split = ids.split()
		img_map[id_split[0]] = int(id_split[1])

	if args.featureType == 'WordsGlove' or args.featureType == 'SentGlove':
		nlp = English()
		print 'loaded word2vec features'
	elif args.featureType == 'BoW':
		num_top_all_words = args.num_top_all_words
		num_top_start_words = args.num_top_start_words
		num_start_words = args.num_start_words
		train_question_file = '../data/preprocessed/questions_train2014.txt'
		vectorizers_list = computeBoWfeatures(num_top_all_words, num_top_start_words, num_start_words, train_question_file)
		# print len(vectorizers_list)
		print 'computed BoW features'
		
	img_dim = 4096
	if args.featureType == 'WordsGlove' or args.featureType == 'SentGlove':
		word_vec_dim = 300
	elif args.featureType == 'BoW':
		word_vec_dim = num_top_all_words + num_top_start_words * num_start_words

	model = Sequential()
	if args.language_only:
		model.add(Dense(args.num_hidden_units, input_dim = word_vec_dim))
	else:
		model.add(Dense(1024, activation = args.activation, input_dim=img_dim + word_vec_dim))
		model.add(Dropout(args.dropout))

	model.add(Dense(1024, activation = args.activation))
	model.add(Dropout(args.dropout))
	model.add(Dense(1024, activation = args.activation))
	model.add(Dropout(args.dropout))

	# for i in xrange(args.num_hidden_layers - 1):
	# 	model.add(Dense(args.num_hidden_units, activation = args.activation))
	# 	if args.dropout > 0:
			# model.add(Dropout(args.dropout))

	model.add(Dense(nb_classes, activation = 'softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	print 'Compilation done...'

	json_string = model.to_json()
	if args.language_only:
		model_file_name = '../models/' + args.featureType + '_mlp_language_only_num_hidden_units_' \
							+ str(args.num_hidden_units) + '_num_hidden_layers_' + str(args.num_hidden_layers)
	else:
		model_file_name = '../models/' + args.featureType + '_mlp_num_hidden_units_' \
							+ str(args.num_hidden_units) + '_num_hidden_layers_' + str(args.num_hidden_layers)
	if args.language_only:
		results_path = '../results/' + args.featureType + '_mlp_language_only_num_hidden_units_' \
							+ str(args.num_hidden_units) + '_num_hidden_layers_' + str(args.num_hidden_layers)
	else:
		results_path = '../results/' + args.featureType + '_mlp_num_hidden_units_' \
							+ str(args.num_hidden_units) + '_num_hidden_layers_' + str(args.num_hidden_layers)		

	open(model_file_name  + '.json', 'w').write(json_string)	

	Acc_train = [0] * args.num_epochs
	Acc_val = [0] * args.num_epochs
	loss_list = [0] * args.num_epochs

	index_shuf = range(len(questions_train))
	shuffle(index_shuf)
	questions_train = [questions_train[i] for i in index_shuf]
	answers_train_all = [answers_train_all[i] for i in index_shuf]
	answers_train = [answers_train[i] for i in index_shuf]	
	images_train = [images_train[i] for i in index_shuf]

	print 'Training started...'

	f1 = open('../results/loss_accuracy_mlp'+ args.featureType +'.txt', 'a')
	f1.write(model_file_name + '\n')

	for k in xrange(args.num_epochs):
		print str(k + 1) + 'th Iteration'
		#shuffle the data points before going through them

		progbar = generic_utils.Progbar(len(questions_train))
		for qu_batch,an_batch,im_batch in zip(grouper(questions_train, args.batch_size, fillvalue=questions_train[-1]), 
											grouper(answers_train, args.batch_size, fillvalue=answers_train[-1]), 
											grouper(images_train, args.batch_size, fillvalue=images_train[-1])):
			if args.featureType == 'WordsGlove':
				X_q_batch = get_questions_matrix_sum(qu_batch, nlp)
			elif args.featureType == 'SentGlove':
				X_q_batch = get_questions_matrix_sentGlove(qu_batch, nlp)
			elif args.featureType == 'BoW':
				X_q_batch = get_questions_BoW(qu_batch, vectorizers_list)
			# print np.shape(X_q_batch)
			if args.language_only:
				X_batch = X_q_batch
			else:
				X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
				X_i_batch_normalized = preprocessing.normalize(X_i_batch, norm='l2')
				X_batch = np.hstack((X_q_batch, X_i_batch_normalized))
		
			Y_batch = get_answers_matrix(an_batch, labelencoder)
			loss = model.train_on_batch(X_batch, Y_batch)
			progbar.add(args.batch_size, values=[("train loss", loss)])
		
		if (k+1) % args.model_save_interval == 0:
			model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k+1))

		loss_list[k] = loss
		f1.write(str(loss_list[k]) + ' ')
		
		# print '	Results on Training set: '
		
		# Acc_train[k] = ValidationMLP(k, model, questions_train, answers_train_all, images_train, img_map, VGGfeatures, labelencoder, \
		# 	 args.batch_size, nb_classes, results_path+'_train', model_file_name+'_train', args.featureType, args.language_only)
		
		print '	Results on Validation set: '

		Acc_val[k] = ValidationMLP(k, model, questions_val, answers_val, images_val, img_map, VGGfeatures, labelencoder, \
			 args.batch_size, nb_classes, results_path, model_file_name, args.featureType, args.language_only)
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
	plt.savefig('../pic/accuracy_train_val' + args.featureType + '.png')

	plt.figure(2)
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.title('Convergence curve')
	plt.plot(range(args.num_epochs), loss_list, 'r--')
	plt.savefig('../pic/Convergence_curve'+ args.featureType + '.png')

	model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k+1))
if __name__ == "__main__":
	main()