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

from utils import grouper, selectFrequentAnswers
from features import *
from progressbar import Bar, ETA, Percentage, ProgressBar    

def Validation_LSTM_embed(k, model, questions, answers, images, img_map ,VGGfeatures, labelencoder, batchSize, nb_classes, results_path, weights_path, vocab_train):

	y_predict_text = []
	
	print 'Validation started...'

	widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'), ' ', ETA()]
	pbar = ProgressBar(widgets=widgets)
	for qu_batch,an_batch,im_batch in pbar(zip(grouper(questions, batchSize, fillvalue=questions[-1]), 
												grouper(answers, batchSize, fillvalue=answers[-1]), 
												grouper(images, batchSize, fillvalue=images[-1]))):

		timesteps = len(qu_batch[-1].split()) #questions sorted in descending order of length
		X_q_batch = get_questions_list_id(qu_batch, vocab_train, timesteps)
		X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
		X_i_batch_normalized = preprocessing.normalize(X_i_batch, norm='l2')
		y_proba = model.predict([X_q_batch, X_i_batch_normalized], verbose=0)
		y_predict = y_proba.argmax(axis = -1)
		# y_predict = keras.np_utils.probas_to_classes(y_proba)
		#print y_predict, y_predict.shape
		y_predict_text.extend(labelencoder.inverse_transform(y_predict))

	correct_val = 0.0
	total = 0	
	binary_correct_val = 0.0
	binary_total = 0.1
	num_correct_val = 0.0
	num_total = 0.1
	other_correct_val = 0.0
	other_total = 0.1
	f1 = open(results_path, 'w')
	for prediction, truth, question, image in zip(y_predict_text, answers, questions, images):
		temp_count=0
		for _truth in truth.split(';'):
			if prediction == _truth:
				temp_count+=1
		if temp_count>2:
			correct_val+=1
		else:
			correct_val+=float(temp_count)/3

		total += 1
		
		binary_temp_count = 0
		num_temp_count = 0
		other_count = 0
		if prediction == 'yes' or prediction == 'no':
			for _truth in truth.split(';'):
				if prediction == _truth:
					binary_temp_count+=1
			if binary_temp_count>2:
				binary_correct_val+=1
			else:
				binary_correct_val+= float(binary_temp_count)/3
			binary_total+=1
		elif np.core.defchararray.isdigit(prediction):
			for _truth in truth.split(';'):
				if prediction == _truth:
					num_temp_count+=1
			if num_temp_count>2:
				num_correct_val+=1
			else:
				num_correct_val+= float(num_temp_count)/3
			num_total+=1
		else:
			for _truth in truth.split(';'):
				if prediction == _truth:
					other_count += 1
			if other_count > 2:
				other_correct_val += 1
			else:
				other_correct_val += float(other_count) / 3
			other_total += 1
 
		f1.write(question.encode('utf-8'))
		f1.write('\n')
		f1.write(image.encode('utf-8'))
		f1.write('\n')
		f1.write(prediction)
		f1.write('\n')
		f1.write(truth.encode('utf-8'))
		f1.write('\n')
		f1.write('\n')

	f1.write('Final Accuracy is ' + str(correct_val/total))
	f1.close()
	f2 = open('../results/overall_results_lstm_embedding.txt', 'a')
	f2.write(str(k) + '\n')
	f2.write(weights_path + '\n')
	f2.write(str(correct_val/total) + '\n\n')
	f2.write(str(binary_correct_val / binary_total) + '\n\n')
	f2.write(str(num_correct_val / num_total) + '\n\n')
	f2.write(str(other_correct_val / other_total) + '\n\n')
	f2.close()

	print 'Final Accuracy is', correct_val/total
	return
	return

def Validation_LSTM_decoder(model, questions, answers, images, img_map ,VGGfeatures, labelencoder, batchSize, nlp, nb_classes, results_path, weights_path):

	nlp = English()

	y_predict_text = []
	
	print 'Validation started...'

	widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'), ' ', ETA()]
	pbar = ProgressBar(widgets=widgets)
	for qu_batch,an_batch,im_batch in pbar(zip(grouper(questions, batchSize, fillvalue=questions[-1]), 
												grouper(answers, batchSize, fillvalue=answers[-1]), 
												grouper(images, batchSize, fillvalue=images[-1]))):

		timesteps = len(nlp(qu_batch[-1])) #questions sorted in descending order of length
		X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, timesteps)
		X_i_batch = get_images_matrix2(im_batch, img_map, VGGfeatures)
		y_proba = model.predict([X_i_batch, X_q_batch, X_i_batch], verbose=0)
		y_predict = y_proba.argmax(axis = -1)
		# y_predict = keras.np_utils.probas_to_classes(y_proba)
		#print y_predict, y_predict.shape
		y_predict_text.extend(labelencoder.inverse_transform(y_predict))

	correct_val = 0.0
	total = 0	
	binary_correct_val = 0.0
	binary_total = 0.1
	num_correct_val = 0.0
	num_total = 0.1
	other_correct_val = 0.0
	other_total = 0.1
	f1 = open(results_path, 'w')
	for prediction, truth, question, image in zip(y_predict_text, answers, questions, images):
		temp_count=0
		for _truth in truth.split(';'):
			if prediction == _truth:
				temp_count+=1
		if temp_count>2:
			correct_val+=1
		else:
			correct_val+=float(temp_count)/3

		total += 1
		
		binary_temp_count = 0
		num_temp_count = 0
		other_count = 0
		if prediction == 'yes' or prediction == 'no':
			for _truth in truth.split(';'):
				if prediction == _truth:
					binary_temp_count+=1
			if binary_temp_count>2:
				binary_correct_val+=1
			else:
				binary_correct_val+= float(binary_temp_count)/3
			binary_total+=1
		elif np.core.defchararray.isdigit(prediction):
			for _truth in truth.split(';'):
				if prediction == _truth:
					num_temp_count+=1
			if num_temp_count>2:
				num_correct_val+=1
			else:
				num_correct_val+= float(num_temp_count)/3
			num_total+=1
		else:
			for _truth in truth.split(';'):
				if prediction == _truth:
					other_count += 1
			if other_count > 2:
				other_correct_val += 1
			else:
				other_correct_val += float(other_count) / 3
			other_total += 1
 
		f1.write(question.encode('utf-8'))
		f1.write('\n')
		f1.write(image.encode('utf-8'))
		f1.write('\n')
		f1.write(prediction)
		f1.write('\n')
		f1.write(truth.encode('utf-8'))
		f1.write('\n')
		f1.write('\n')

	f1.write('Final Accuracy is ' + str(correct_val/total))
	f1.close()
	f2 = open('../results/overall_results_lstm.txt', 'a')
	f2.write(weights_path + '\n')
	f2.write(str(correct_val/total) + '\n\n')
	f2.write(str(binary_correct_val / binary_total) + '\n\n')
	f2.write(str(num_correct_val / num_total) + '\n\n')
	f2.write(str(other_correct_val / other_total) + '\n\n')
	f2.close()

	print 'Final Accuracy is', correct_val/total
	return

def ValidationFromFile(questions, answers, images, img_map ,VGGfeatures, labelencoder, batchSize, nlp, nb_classes, results_path, weights_path):
	parser = argparse.ArgumentParser()
	parser.add_argument('-model', type=str, required=False)
	parser.add_argument('-weights', type=str, required=False) 
	parser.add_argument('-results', type=str, required=False)
	args = parser.parse_args()

	args.model = '../models/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_num_hidden_layers_lstm_1.json'
	args.weights = '../models/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_num_hidden_layers_lstm_1_epoch_080.hdf5'

	model = model_from_json(open(args.model).read())
	model.load_weights(args.weights)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	y_predict_text = []

	print 'Validation started...'

	widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'), ' ', ETA()]
	pbar = ProgressBar(widgets=widgets)
	for qu_batch,an_batch,im_batch in pbar(zip(grouper(questions, batchSize, fillvalue=questions[0]), 
												grouper(answers, batchSize, fillvalue=answers[0]), 
												grouper(images, batchSize, fillvalue=images[0]))):
		timesteps = len(nlp(qu_batch[-1])) #questions sorted in descending order of length
		X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, timesteps)
		X_i_batch = get_images_matrix2(im_batch, img_map, VGGfeatures)
		y_proba = model.predict([X_i_batch, X_q_batch], verbose=0)
		y_predict = y_proba.argmax(axis = -1)
		# y_predict = keras.np_utils.probas_to_classes(y_proba)
		#print y_predict, y_predict.shape
		y_predict_text.extend(labelencoder.inverse_transform(y_predict))

	total = 0
	correct_val=0.0
	f1 = open(results_path, 'w')
	for prediction, truth, question, image in zip(y_predict_text, answers, questions, images):
		temp_count=0
		for _truth in truth.split(';'):
			if prediction == _truth:
				temp_count+=1

		if temp_count>2:
			correct_val+=1
		else:
			correct_val+=float(temp_count)/3

		correct_val = 0.0
	total = 0	
	binary_correct_val = 0.0
	binary_total = 0.1
	num_correct_val = 0.0
	num_total = 0.1
	other_correct_val = 0.0
	other_total = 0.1
	f1 = open(results_path, 'w')
	for prediction, truth, question, image in zip(y_predict_text, answers, questions, images):
		temp_count=0
		for _truth in truth.split(';'):
			if prediction == _truth:
				temp_count+=1
		if temp_count>2:
			correct_val+=1
		else:
			correct_val+=float(temp_count)/3

		total += 1
		
		binary_temp_count = 0
		num_temp_count = 0
		other_count = 0
		if prediction == 'yes' or prediction == 'no':
			for _truth in truth.split(';'):
				if prediction == _truth:
					binary_temp_count+=1
			if binary_temp_count>2:
				binary_correct_val+=1
			else:
				binary_correct_val+= float(binary_temp_count)/3
			binary_total+=1
		elif np.core.defchararray.isdigit(prediction):
			for _truth in truth.split(';'):
				if prediction == _truth:
					num_temp_count+=1
			if num_temp_count>2:
				num_correct_val+=1
			else:
				num_correct_val+= float(num_temp_count)/3
			num_total+=1
		else:
			for _truth in truth.split(';'):
				if prediction == _truth:
					other_count += 1
			if other_count > 2:
				other_correct_val += 1
			else:
				other_correct_val += float(other_count) / 3
			other_total += 1
 		
		f1.write(question.encode('utf-8'))
		f1.write('\n')
		f1.write(image.encode('utf-8'))
		f1.write('\n')
		f1.write(prediction)
		f1.write('\n')
		f1.write(truth.encode('utf-8'))
		f1.write('\n')
		f1.write('\n')

	f1.write('Final Accuracy is ' + str(correct_val/total))
	f1.close()
	f2 = open('../results/overall_results_lstm_decoder.txt', 'a')
	f2.write(weights_path + '\n')
	f2.write(str(correct_val/total) + '\n\n')
	f2.write(str(binary_correct_val / binary_total) + '\n\n')
	f2.write(str(num_correct_val / num_total) + '\n\n')
	f2.write(str(other_correct_val / other_total) + '\n\n')
	f2.close()
	print 'Final Accuracy is', correct_val/total
	return

def Validation_LSTM_encoder(model, questions, answers, images, img_map ,VGGfeatures, labelencoder, batchSize, nlp, nb_classes, results_path, weights_path):

	nlp = English()

	y_predict_text = []
	
	print 'Validation started...'

	widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'), ' ', ETA()]
	pbar = ProgressBar(widgets=widgets)
	for qu_batch,an_batch,im_batch in pbar(zip(grouper(questions, batchSize, fillvalue=questions[-1]), 
												grouper(answers, batchSize, fillvalue=answers[-1]), 
												grouper(images, batchSize, fillvalue=images[-1]))):

		timesteps = len(nlp(qu_batch[-1])) #questions sorted in descending order of length
		X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, timesteps)
		X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
		X_i_batch_normalized = preprocessing.normalize(X_i_batch, norm='l2')
		#print X_i_batch.shape, X_q_batch.shape
		# Y_batch = get_answers_matrix(an_batch, labelencoder)
		y_proba = model.predict([X_q_batch, X_i_batch_normalized], verbose=0)
		y_predict = y_proba.argmax(axis = -1)
		# y_predict = keras.np_utils.probas_to_classes(y_proba)
		#print y_predict, y_predict.shape
		y_predict_text.extend(labelencoder.inverse_transform(y_predict))

	correct_val = 0.0
	total = 0	
	binary_correct_val = 0.0
	binary_total = 0.1	
	num_correct_val = 0.0
	num_total = 0.1
	other_correct_val = 0.0
	other_total = 0.1
	f1 = open(results_path, 'w')
	for prediction, truth, question, image in zip(y_predict_text, answers, questions, images):
		temp_count=0
		for _truth in truth.split(';'):
			if prediction == _truth:
				temp_count+=1
		if temp_count>2:
			correct_val+=1
		else:
			correct_val+=float(temp_count)/3

		total += 1
		
		binary_temp_count = 0
		num_temp_count = 0
		other_count = 0
		if prediction == 'yes' or prediction == 'no':
			for _truth in truth.split(';'):
				if prediction == _truth:
					binary_temp_count+=1
			if binary_temp_count>2:
				binary_correct_val+=1
			else:
				binary_correct_val+= float(binary_temp_count)/3
			binary_total+=1
		elif np.core.defchararray.isdigit(prediction):
			for _truth in truth.split(';'):
				if prediction == _truth:
					num_temp_count+=1
			if num_temp_count>2:
				num_correct_val+=1
			else:
				num_correct_val+= float(num_temp_count)/3
			num_total+=1
		else:
			for _truth in truth.split(';'):
				if prediction == _truth:
					other_count += 1
			if other_count > 2:
				other_correct_val += 1
			else:
				other_correct_val += float(other_count) / 3
			other_total += 1
 
		f1.write(question.encode('utf-8'))
		f1.write('\n')
		f1.write(image.encode('utf-8'))
		f1.write('\n')
		f1.write(prediction)
		f1.write('\n')
		f1.write(truth.encode('utf-8'))
		f1.write('\n')
		f1.write('\n')

	f1.write('Final Accuracy is ' + str(correct_val/total))
	f1.close()
	f2 = open('../results/overall_results_lstm_encoder.txt', 'a')
	f2.write(weights_path + '\n')
	f2.write(str(correct_val/total) + '\n\n')
	f2.write(str(binary_correct_val / binary_total) + '\n\n')
	f2.write(str(num_correct_val / num_total) + '\n\n')
	f2.write(str(other_correct_val / other_total) + '\n\n')
	f2.close()

	print 'Final Accuracy is', correct_val/total
	return correct_val/total

def ValidationMLP(k, model, questions, answers, images, img_map ,VGGfeatures, labelencoder, batchSize, nb_classes, results_path, weights_path, featureType, language_only):

	if featureType == 'WordsGlove' or featureType == 'SentGlove':
		nlp = English()
	elif featureType == 'BoW':
		num_top_all_words = 1000
		num_top_start_words = 10
		num_start_words = 3
		train_question_file = '../data/preprocessed/questions_train2014.txt'
		vectorizers_list = computeBoWfeatures(num_top_all_words, num_top_start_words, num_start_words, train_question_file)

	y_predict_text = []
	
	print 'Validation started...'

	widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'), ' ', ETA()]
	pbar = ProgressBar(widgets=widgets)
	for qu_batch,an_batch,im_batch in pbar(zip(grouper(questions, batchSize, fillvalue=questions[-1]), 
												grouper(answers, batchSize, fillvalue=answers[-1]), 
												grouper(images, batchSize, fillvalue=images[-1]))):

		if featureType == 'WordsGlove':
			X_q_batch = get_questions_matrix_sum(qu_batch, nlp)
		elif featureType == 'SentGlove':
			X_q_batch = get_questions_matrix_sentGlove(qu_batch, nlp)
		elif featureType == 'BoW':
			X_q_batch = get_questions_BoW(qu_batch, vectorizers_list)
		# print np.shape(X_q_batch)
		if language_only == True:
			X_batch = X_q_batch
		else:
			X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
			X_i_batch_normalized = preprocessing.normalize(X_i_batch, norm='l2')
			X_batch = np.hstack((X_q_batch, X_i_batch_normalized))

		y_proba = model.predict(X_batch, verbose=0)
		y_predict = y_proba.argmax(axis = -1)
		# y_predict = keras.np_utils.probas_to_classes(y_proba)
		#print y_predict, y_predict.shape
		y_predict_text.extend(labelencoder.inverse_transform(y_predict))

	correct_val = 0.0
	total = 0	
	binary_correct_val = 0.0
	binary_total = 0.1
	num_correct_val = 0.0
	num_total = 0.1
	other_correct_val = 0.0
	other_total = 0.1
	f1 = open(results_path, 'w')
	for prediction, truth, question, image in zip(y_predict_text, answers, questions, images):
		temp_count=0
		for _truth in truth.split(';'):
			if prediction == _truth:
				temp_count+=1
		if temp_count>2:
			correct_val+=1
		else:
			correct_val+=float(temp_count)/3

		total += 1
		
		binary_temp_count = 0
		num_temp_count = 0
		other_count = 0
		if prediction == 'yes' or prediction == 'no':
			for _truth in truth.split(';'):
				if prediction == _truth:
					binary_temp_count+=1
			if binary_temp_count>2:
				binary_correct_val+=1
			else:
				binary_correct_val+= float(binary_temp_count)/3
			binary_total+=1
		elif np.core.defchararray.isdigit(prediction):
			for _truth in truth.split(';'):
				if prediction == _truth:
					num_temp_count+=1
			if num_temp_count>2:
				num_correct_val+=1
			else:
				num_correct_val+= float(num_temp_count)/3
			num_total+=1
		else:
			for _truth in truth.split(';'):
				if prediction == _truth:
					other_count += 1
			if other_count > 2:
				other_correct_val += 1
			else:
				other_correct_val += float(other_count) / 3
			other_total += 1
 
		f1.write(question.encode('utf-8'))
		f1.write('\n')
		f1.write(image.encode('utf-8'))
		f1.write('\n')
		f1.write(prediction)
		f1.write('\n')
		f1.write(truth.encode('utf-8'))
		f1.write('\n')
		f1.write('\n')

	f1.write('Final Accuracy is ' + str(correct_val/total))
	f1.close()
	f2 = open('../results/overall_results_mlp_' + featureType + '.txt', 'a')
	f2.write(str(k) + '\n')
	f2.write(weights_path + '\n')
	f2.write(str(correct_val/total) + '\n\n')
	f2.write(str(binary_correct_val / binary_total) + '\n\n')
	f2.write(str(num_correct_val / num_total) + '\n\n')
	f2.write(str(other_correct_val / other_total) + '\n\n')
	f2.close()

	print 'Final Accuracy is', correct_val/total

	return correct_val/total

def ValidationMLP_MidLayer(model, questions, answers, images, img_map ,VGGfeatures, labelencoder, batchSize, nb_classes, results_path, weights_path, featureType):

	if featureType == 'WordsGlove' or featureType == 'SentGlove':
		nlp = English()
	elif featureType == 'BoW':
		num_top_all_words = 1000
		num_top_start_words = 10
		num_start_words = 3
		train_question_file = '../data/preprocessed/questions_train2014.txt'
		vectorizers_list = computeBoWfeatures(num_top_all_words, num_top_start_words, num_start_words, train_question_file)

	y_predict_text = []
	
	print 'Validation started...'

	widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'), ' ', ETA()]
	pbar = ProgressBar(widgets=widgets)
	for qu_batch,an_batch,im_batch in pbar(zip(grouper(questions, batchSize, fillvalue=questions[-1]), 
												grouper(answers, batchSize, fillvalue=answers[-1]), 
												grouper(images, batchSize, fillvalue=images[-1]))):

		if featureType == 'WordsGlove':
			X_q_batch = get_questions_matrix_sum(qu_batch, nlp)
		elif featureType == 'SentGlove':
			X_q_batch = get_questions_matrix_sentGlove(qu_batch, nlp)
		elif featureType == 'BoW':
			X_q_batch = get_questions_BoW(qu_batch, vectorizers_list)
		# print np.shape(X_q_batch)

		X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
		X_i_batch_normalized = preprocessing.normalize(X_i_batch, norm='l2')

		y_proba = model.predict([X_i_batch_normalized, X_q_batch], verbose=0)
		y_predict = y_proba.argmax(axis = -1)
		# y_predict = keras.np_utils.probas_to_classes(y_proba)
		#print y_predict, y_predict.shape
		y_predict_text.extend(labelencoder.inverse_transform(y_predict))

	correct_val = 0.0
	total = 0	
	binary_correct_val = 0.0
	binary_total = 0	
	num_correct_val = 0.0
	num_total = 0
	other_correct_val = 0.0
	other_total = 0
	f1 = open(results_path, 'w')
	for prediction, truth, question, image in zip(y_predict_text, answers, questions, images):
		temp_count=0
		for _truth in truth.split(';'):
			if prediction == _truth:
				temp_count+=1
		if temp_count>2:
			correct_val+=1
		else:
			correct_val+=float(temp_count)/3

		total += 1
		
		binary_temp_count = 0
		num_temp_count = 0
		other_count = 0
		if prediction == 'yes' or prediction == 'no':
			for _truth in truth.split(';'):
				if prediction == _truth:
					binary_temp_count+=1
			if binary_temp_count>2:
				binary_correct_val+=1
			else:
				binary_correct_val+= float(binary_temp_count)/3
			binary_total+=1
		elif np.core.defchararray.isdigit(prediction):
			for _truth in truth.split(';'):
				if prediction == _truth:
					num_temp_count+=1
			if num_temp_count>2:
				num_correct_val+=1
			else:
				num_correct_val+= float(num_temp_count)/3
			num_total+=1
		else:
			for _truth in truth.split(';'):
				if prediction == _truth:
					other_count += 1
			if other_count > 2:
				other_correct_val += 1
			else:
				other_correct_val += float(other_count) / 3
			other_total += 1
 
		f1.write(question.encode('utf-8'))
		f1.write('\n')
		f1.write(image.encode('utf-8'))
		f1.write('\n')
		f1.write(prediction)
		f1.write('\n')
		f1.write(truth.encode('utf-8'))
		f1.write('\n')
		f1.write('\n')

	f1.write('Final Accuracy is ' + str(correct_val/total))
	f1.close()
	f2 = open('../results/overall_results_mlp_midlayer.txt', 'a')
	f2.write(weights_path + '\n')
	f2.write(str(correct_val/total) + '\n\n')
	f2.write(str(binary_correct_val / binary_total) + '\n\n')
	f2.write(str(num_correct_val / num_total) + '\n\n')
	f2.write(str(other_correct_val / other_total) + '\n\n')
	f2.close()

	print 'Final Accuracy is', correct_val/total
	return correct_val/total