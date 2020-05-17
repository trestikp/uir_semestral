from math import log
import numpy as np

def compute_idfs(c_vects):
	vocab = []
	vocab = np.zeros(len(c_vects[list(c_vects.keys())[0]]), dtype=int)
	for c in c_vects:
		for w in range(len(c_vects[c])):
			if c_vects[c][w] > 0:
				vocab[w] += 1
	for w in vocab:
		try:
			vocab[w] = log(len(c_vects) / float(vocab[w]))
		except ZeroDivisionError:
			vocab[w] = 0
	return vocab
	"""
	vocab = dict.fromkeys(vocabulary, 0)
	for c in c_vects:
		for w in vocabulary:
			try:
				if c_vects[vocabulary.index(w)] > 0:
					print(vocab[w])
					vocab[w] += 1
			except KeyError:
				vocab[w] = 1
				#print(f"Word {w} is not in vocabulary. Skipping word...")
	for w in vocab:
		try:
			vocab[w] = log(len(c_vects) / float(vocab[w]))
		except ZeroDivisionError:
			#this is usuallly prevented by +1 to numerator
			vocab[w] = 0
	return vocab
	"""

def compute_tf(word_count, class_wc):
	return word_count / float(class_wc)

def calculate_tf_idf(model):
	idfs = compute_idfs(model[2])
	for c in model[2]:
		for w in model[2][c]:
			tf = compute_tf(model[2][c][w], model[1][c])
			model[2][c][w] = tf * idfs[w]
	return model

def total_number_of_words(c_wcount):
	total = 0
	for c in c_wcount:
		total += c_wcount[c]
	return total

def normalize_tf_idf(model):
	total = total_number_of_words(model[1])
	for c in model[2]:
		for w in model[2][c]:
			model[2][c][w] = model[2][c][w] * total
	return model
