from math import log
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

def compute_idfs(c_dicts):
	vocab = set()
	for c in c_dicts:
		vocab |= set(c_dicts[c])
	vocab = dict.fromkeys(vocab, 0)
	for c in c_dicts:
		for w in c_dicts[c]:
			try:
				vocab[w] += 1
			except KeyError:
				print("Word not in vocabulary. Skipping word...")
	for w in vocab:
		try:
			vocab[w] = log((len(c_dicts)) / (float(vocab[w])))
		except ZeroDivisionError:
			vocab[w] = 0
	return vocab
	

def compute_tf(word_count, class_wc):
	return word_count / float(class_wc)

def calculate_tf_idf(model):
	idfs = compute_idfs(model[2])
	for c in model[2]:
		for w in model[2][c]:
			tf = compute_tf(model[2][c][w], model[1][c])
			model[2][c][w] = tf * idfs[w]
	return model

def total_number_of_word(c_wcount):
	total = 0
	for c in c_wcount:
		total += c_wcount[c]
	return total

def normalize_tf_idf(model):
	tot = total_number_of_word(model[1])
	for c in model[2]:
		for w in model[2][c]:
			model[2][c][w] = model[2][c][w] * tot
	return model

if __name__ == "__main__":
	#documentA = 'the man went out for a walk'
	#documentB = 'the children sat around the fire'

	#one = "tohe je tf idf pokus"
	#two = "timto je tf idf hokus"
	fake_model = [{"one": 1,"two": 1},\
		      {"one": 7,"two": 6},\
		      {"one": {"the": 1, "man": 1, "went": 1, "out": 1,\
		      	       "for": 1, "a": 1, "walk": 1},\
		       "two": {"the": 2, "children": 1, "sat": 1, "around": 1,\
		       	       "fire": 1}}]
	print(f"before tfidf: {fake_model}\n")
	calculate_tf_idf(fake_model)
	print(f"after tfidf: {fake_model}\n")
	normalize_tf_idf(fake_model)
	print(f"normalized tfidf: {fake_model}\n")
