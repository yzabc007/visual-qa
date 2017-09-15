import operator
from itertools import izip_longest
from collections import defaultdict

def selectFrequentAnswers(questions_train, answers_train, answers_train_all, images_train, maxAnswers):
	answer_fq= defaultdict(int)
	#build a dictionary of answers
	for answer in answers_train:
		answer_fq[answer] += 1

	sorted_fq = sorted(answer_fq.items(), key=operator.itemgetter(1), reverse=True)[0:maxAnswers]
	top_answers, top_fq = zip(*sorted_fq)
	new_one_answer_train=[]
	new_answers_train_all = []
	new_questions_train=[]
	new_images_train=[]
	#only those answer which appear int he top 1K are used for training
	for answer,answer_all, question,image in zip(answers_train, answers_train_all, questions_train, images_train):
		if answer in top_answers:
			new_one_answer_train.append(answer)
			new_answers_train_all.append(answer_all)
			new_questions_train.append(question)
			new_images_train.append(image)

	return (new_questions_train, new_one_answer_train, new_answers_train_all, new_images_train)

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)