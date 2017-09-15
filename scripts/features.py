import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

import re

def get_questions_list_id(questions, vocab_train, timesteps):
	assert not isinstance(questions, basestring)
	nb_samples = len(questions)
	questions_list = np.zeros((nb_samples, timesteps))
	for i in xrange(len(questions)):
		q = questions[i]
		q = re.sub('[\W]+', ' ', q.lower())
		q = q.split()
		for j in xrange(len(q)):
			if j < timesteps:
				questions_list[i, j] = vocab_train.GetID(q[j])
	
	return questions_list

def get_questions_tensor_timeseries(questions, nlp, timesteps):
	'''
	Returns a time series of word vectors for tokens in the question

	Input:
	questions: list of unicode objects
	nlp: an instance of the class English() from spacy.en
	timesteps: the number of 

	Output:
	A numpy ndarray of shape: (nb_samples, timesteps, word_vec_dim)
	'''
	assert not isinstance(questions, basestring)
	nb_samples = len(questions)
	word_vec_dim = nlp(questions[0])[0].vector.shape[0]
	questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
	for i in xrange(len(questions)):
		q = questions[i].lower()
		q = re.sub('[\W]+', ' ', q)
		tokens = nlp(q)
		for j in xrange(len(tokens)):
			if j < timesteps:
				#v = preprocessing.normalize(tokens[j].vector.reshape(-1, 1), norm='l2')
				#questions_tensor[i,j,:] = v.reshape(1, -1)
				questions_tensor[i,j,:] = tokens[j].vector

	return questions_tensor

def get_questions_matrix_sum(questions, nlp):
	'''
	Sums the word vectors of all the tokens in a question
	
	Input:
	questions: list of unicode objects
	nlp: an instance of the class English() from spacy.en

	Output:
	A numpy array of shape: (nb_samples, word_vec_dim)	
	'''
	assert not isinstance(questions, basestring)
	nb_samples = len(questions)
	word_vec_dim = nlp(questions[0])[0].vector.shape[0]
	questions_matrix = np.zeros((nb_samples, word_vec_dim))
	for i in xrange(len(questions)):
		tokens = nlp(questions[i])
		for j in xrange(len(tokens)):
			questions_matrix[i,:] += tokens[j].vector
			
		#questions_matrix[i,:] = questions_matrix[i,:]/len(tokens)
		
	return questions_matrix

def get_questions_matrix_average(questions, nlp):

	assert not isinstance(questions, basestring)
	nb_samples = len(questions)
	word_vec_dim = nlp(questions[0])[0].vector.shape[0]
	questions_matrix = np.zeros((nb_samples, word_vec_dim))
	for i in xrange(len(questions)):
		tokens = nlp(questions[i])
		for j in xrange(len(tokens)):
			questions_matrix[i,:] += tokens[j].vector
		questions_matrix[i,:] = questions_matrix[i,:]/len(tokens)
		
	return questions_matrix

def get_questions_matrix_sentGlove(questions, nlp):

	assert not isinstance(questions, basestring)
	nb_samples = len(questions)
	word_vec_dim = nlp(questions[0]).vector.shape[0]
	questions_matrix = np.zeros((nb_samples, word_vec_dim))
	for i in xrange(nb_samples):
		questions_matrix[i, :] = nlp(questions[i]).vector
	return questions_matrix

def computeBoWfeatures(num_top_all_words, num_top_start_words, num_start_words, train_question_file):
	# num_top_all_words = 1000
	# num_top_start_words = 10
	# train_question = open('../data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
	train_question = open(train_question_file, 'r').read().decode('utf8').splitlines()
	corpus_all = train_question
	#top 1000 words in question_train
	vectorizers_list = []
	vectorizer_all = CountVectorizer(analyzer = "word",   \
								 tokenizer = None,    \
								 preprocessor = None, \
								 stop_words = None,   \
								 max_features = num_top_all_words) 
	vectorizer_all.fit_transform(corpus_all)
	vectorizers_list.append(vectorizer_all)

	if num_top_start_words != 0:
		corpus_start = []
		for i in range(num_start_words):
			for j in range(len(train_question)):
				questions_in_words = train_question[j].split()
				if len(questions_in_words) >= i + 1:
					corpus_start.append(questions_in_words[i])

			vectorizer_start = CountVectorizer(analyzer = "word",   \
								 tokenizer = None,    \
								 preprocessor = None, \
								 stop_words = None,   \
								 max_features = num_top_start_words)
			vectorizer_start.fit_transform(corpus_start)
			vectorizers_list.append(vectorizer_start)

		# corpus_1 = []
		# corpus_2 = []
		# corpus_3 = []
		# for i in range(len(train_question)):
		# 	questions_in_words = train_question[i].split()
		# 	if len(questions_in_words) >= 1:
		# 		corpus_1.append(questions_in_words[0])
		# 	if len(questions_in_words) >= 2:
		# 		corpus_2.append(questions_in_words[1])
		# 	if len(questions_in_words) >= 3:
		# 		corpus_3.append(questions_in_words[2])
		# #top 10 first word
		# vectorizer_1 = CountVectorizer(analyzer = "word",   \
		# 							 tokenizer = None,    \
		# 							 preprocessor = None, \
		# 							 stop_words = None,   \
		# 							 max_features = num_top_start_words)	
		# vectorizer_1.fit_transform(corpus_1)
		# #top 10 second word
		# vectorizer_2 = CountVectorizer(analyzer = "word",   \
		# 							 tokenizer = None,    \
		# 							 preprocessor = None, \
		# 							 stop_words = None,   \
		# 							 max_features = num_top_start_words)	
		# vectorizer_2.fit_transform(corpus_2)
		# #top 10 third word
		# vectorizer_3 = CountVectorizer(analyzer = "word",   \
		# 							 tokenizer = None,    \
		# 							 preprocessor = None, \
		# 							 stop_words = None,   \
		# 							 max_features = num_top_start_words)	
		# vectorizer_3.fit_transform(corpus_3)

	return vectorizers_list

def get_questions_BoW(questions, vectorizers_list):

	questions_BoW_features = vectorizers_list[0].transform(questions).toarray()
	
	for i in range(1, len(vectorizers_list)):
		temp_qu_features = vectorizers_list[i].transform(questions).toarray()
		questions_BoW_features = np.hstack((questions_BoW_features, temp_qu_features))

	# first_qu_features = vectorizer_1.transform(questions).toarray()
	# questions_BoW_features = np.hstack((questions_BoW_features, first_qu_features))
	# #
	# second_qu_features = vectorizer_2.transform(questions).toarray()
	# questions_BoW_features = np.hstack((questions_BoW_features, second_qu_features))
	# #
	# third_qu_features = vectorizer_3.transform(questions).toarray()
	# questions_BoW_features = np.hstack((questions_BoW_features, third_qu_features))

	return questions_BoW_features
	
# def get_questions_BoW_features(questions, num_top_all_words, num_top_start_words):

# 	#print "Creating the bag of words...\n"
	
# 	questions_train = open('../data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
	
# 	#questions_train = [
# 	#	'This word the first document.',
# 	#	'The sun the second second document.',
# 	#	'Is sea third one.',
# 	#	'Is day the first document?',
# 	#]
# 	corpus = questions_train
	
# 	# Initialize the "CountVectorizer" object, which is scikit-learn's
# 	# bag of words tool.  
# 	vectorizer = CountVectorizer(analyzer = "word",   \
# 								 tokenizer = None,    \
# 								 preprocessor = None, \
# 								 stop_words = None,   \
# 								 max_features = num_top_all_words) 
								
# 	# fit_transform() does two functions: First, it fits the model
# 	# and learns the vocabulary; second, it transforms our training data
# 	# into feature vectors. The input to fit_transform should be a list of 
# 	# strings.
# 	vectorizer.fit_transform(corpus)
# 	questions_BoW_features = vectorizer.transform(questions).toarray()
# 	#print(vectorizer.get_feature_names())
	
# 	if num_top_start_words != 0:
# 		corpus_1 = []
# 		corpus_2 = []
# 		corpus_3 = []
# 		for i in range(len(questions_train)):
# 			questions_in_words = questions_train[i].split()
# 			if len(questions_in_words) >= 1:
# 				corpus_1.append(questions_in_words[0])
# 			if len(questions_in_words) >= 2:
# 				corpus_2.append(questions_in_words[1])
# 			if len(questions_in_words) >= 3:
# 				corpus_3.append(questions_in_words[2])
		
# 		vectorizer2 = CountVectorizer(analyzer = "word",   \
# 									 tokenizer = None,    \
# 									 preprocessor = None, \
# 									 stop_words = None,   \
# 									 max_features = num_top_start_words) 
		
# 		#train new vectorizer for first top words			 
# 		vectorizer2.fit_transform(corpus_1)
# 		#print(vectorizer2.get_feature_names())
# 		first_qu_features = vectorizer2.transform(questions).toarray()
# 		questions_BoW_features = np.hstack((questions_BoW_features, first_qu_features))
		
# 		#train new vectorizer for second top words
# 		vectorizer2.fit_transform(corpus_2)
# 		#print(vectorizer2.get_feature_names())
# 		second_qu_features = vectorizer2.transform(questions).toarray()
# 		questions_BoW_features = np.hstack((questions_BoW_features, second_qu_features))
		
# 		#train new vectorizer for third top words
# 		vectorizer2.fit_transform(corpus_3)
# 		#print(vectorizer2.get_feature_names())
# 		third_qu_features = vectorizer2.transform(questions).toarray()
# 		questions_BoW_features = np.hstack((questions_BoW_features, third_qu_features))
	
# 	return questions_BoW_features
	
def get_answers_matrix(answers, encoder):
	'''
	Converts string objects to class labels

	Input:
	answers:	a list of unicode objects
	encoder:	a scikit-learn LabelEncoder object

	Output:
	A numpy array of shape (nb_samples, nb_classes)
	'''
	assert not isinstance(answers, basestring)
	y = encoder.transform(answers) #string to numerical class [0, 1, 1, 3, 5 ...] -> [[1, 0, ..], [1, 0, ...], ...]
	nb_classes = encoder.classes_.shape[0]
	Y = np.zeros((len(y), nb_classes))
	for i in range(len(y)):
		Y[i, y[i]] = 1.
	return Y

def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
	'''
	Gets the 4096-dimensional CNN features for the given COCO
	images
	
	Input:
	img_coco_ids: 	A list of strings, each string corresponding to
				  	the MS COCO Id of the relevant image
	img_map: 		A dictionary that maps the COCO Ids to their indexes 
					in the pre-computed VGG features matrix
	VGGfeatures: 	A numpy array of shape (nb_dimensions,nb_images)

	Ouput:
	A numpy matrix of size (nb_samples, nb_dimensions)
	'''
	assert not isinstance(img_coco_ids, basestring)
	nb_samples = len(img_coco_ids)
	nb_dimensions = VGGfeatures.shape[0]
	image_matrix = np.zeros((nb_samples, nb_dimensions))
	for j in xrange(len(img_coco_ids)):
		image_matrix[j,:] = VGGfeatures[:,img_map[img_coco_ids[j]]]

	return image_matrix

def get_images_matrix2(img_coco_ids, img_map, VGGfeatures):
	'''
	Gets the 4096-dimensional CNN features for the given COCO
	images
	
	Input:
	img_coco_ids: 	A list of strings, each string corresponding to
				  	the MS COCO Id of the relevant image
	img_map: 		A dictionary that maps the COCO Ids to their indexes 
					in the pre-computed VGG features matrix
	VGGfeatures: 	A numpy array of shape (nb_dimensions,nb_images)

	Ouput:
	A numpy matrix of size (nb_samples, nb_dimensions)
	'''
	assert not isinstance(img_coco_ids, basestring)
	nb_samples = len(img_coco_ids)
	nb_dimensions = VGGfeatures.shape[0]
	image_matrix = np.zeros((nb_samples, 1, nb_dimensions))
	for j in xrange(len(img_coco_ids)):
		image_matrix[j, 0, :] = preprocessing.normalize(VGGfeatures[:,img_map[img_coco_ids[j]]].reshape(1, -1), norm = 'l2')

	return image_matrix