import numpy as np
import time
import random
from hmm import HMM


def accuracy(predict_tagging, true_tagging):
	if len(predict_tagging) != len(true_tagging):
		return 0, 0, 0
	cnt = 0
	for i in range(len(predict_tagging)):
		if predict_tagging[i] == true_tagging[i]:
			cnt += 1
	total_correct = cnt
	total_words = len(predict_tagging)
	if total_words == 0:
		return 0, 0, 0
	return total_correct, total_words, total_correct*1.0/total_words


class Dataset:

	def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
		tags = self.read_tags(tagfile)
		data = self.read_data(datafile)
		self.tags = tags
		lines = []
		for l in data:
			new_line = self.Line(l)
			if new_line.length > 0:
				lines.append(new_line)
		if seed is not None: random.seed(seed)
		random.shuffle(lines)
		train_size = int(train_test_split * len(data))
		self.train_data = lines[:train_size]
		self.test_data = lines[train_size:]
		return

	def read_data(self, filename):
		"""Read tagged sentence data"""
		with open(filename, 'r') as f:
			sentence_lines = f.read().split("\n\n")
		return sentence_lines

	def read_tags(self, filename):
		"""Read a list of word tag classes"""
		with open(filename, 'r') as f:
			tags = f.read().split("\n")
		return tags

	class Line:
		def __init__(self, line):
			words = line.split("\n")
			self.id = words[0]
			self.words = []
			self.tags = []

			for idx in range(1, len(words)):
				pair = words[idx].split("\t")
				self.words.append(pair[0])
				self.tags.append(pair[1])
			self.length = len(self.words)
			return

		def show(self):
			print(self.id)
			print(self.length)
			print(self.words)
			print(self.tags)
			return

def getUniqueDict(body):
	#wordDict = {}
	wordDict = dict()
	index = 0
	for sentence in body:
		for word in sentence.words:
			if word not in wordDict:
				wordDict[word] = index
				index += 1
	#print("unique words:")
	#print(len(wordDict))
	return wordDict

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	num_sentence = len(train_data)
	num_tags = len(tags)

	for d in train_data:
		#print(d.tags)
		#print()
		break

	state_dict = {i:j for (i,j) in zip(tags, range(num_tags))}

	obs_dict = getUniqueDict(train_data)
	#print(state_dict)

	pi = np.zeros(num_tags)
	for d in train_data:
		pos = d.tags[0]
		pos_id = state_dict[pos]
		pi[pos_id] += 1
	pi /= num_sentence

	A = np.zeros((num_tags, num_tags))
	
	for sentence in train_data:
		for i, w in enumerate(sentence.tags[:len(sentence.tags) - 1]):
			#A[state_dict[w]][state_dict[sentence.tags[i + 1]]] += 1
			curr_state = state_dict[w]
			next_state = state_dict[sentence.tags[i + 1]]
			A[curr_state][next_state] += 1
	
	"""
	for sentence in train_data:
		for (i,j) in zip(sentence.tags,sentence.tags[1:]):
			state_i = state_dict[i]
			state_j = state_dict[j]
			A[state_i][state_j] += 1
	"""

	for row in A:
		s = sum(row)
		if s > 0:
			row /= s


	B = np.zeros((num_tags, len(obs_dict)))
	for sentence in train_data:
		for i, w in enumerate(sentence.tags):
			word = sentence.words[i]
			s = state_dict[w]
			o = obs_dict[word]
			B[s][o] += 1

	for row in B:
		s = sum(row)
		if s > 0:
			row /= s

	model = HMM(pi, A, B, obs_dict, state_dict)
	###################################################
	return model


# TODO:
def speech_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	num_sentence = len(test_data)
	num_tags = len(tags)

	tagging = []
	for i, d in enumerate(test_data):
		#tagging.append(d.tags)
		for w in d.words:
			if w not in model.obs_dict:
				model.obs_dict[w] = len(model.obs_dict)
				emissions_col = np.full((num_tags, 1), 1**-6)
				model.B = np.hstack((model.B, emissions_col))
		path = model.viterbi(d.words)
		tagging.append(path)
	###################################################
	return tagging

