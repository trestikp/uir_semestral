import re
import glob
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import partial


def extract_words(article):
	article = re.sub("[^\w]", " ", article).split()
	return [word.lower().strip() for word in article]

def extract_annotations(line):
	line = line.split()
	return [l.strip() for l in line]

def extract_bigrams_from_corpus(parsed_file_data):
	bigrams = {}
	if len(parsed_file_data) > 0:
		bigrams[parsed_file_data[0]] = 1
		for d in range(1, len(parsed_file_data)):
			try:
				bigrams[parsed_file_data[d - 1] + " " + parsed_file_data[d]] += 1
			except KeyError:
				bigrams[parsed_file_data[d - 1] + " " + parsed_file_data[d]] = 1
		bigrams[parsed_file_data[len(parsed_file_data) - 1]] = 1
	return bigrams

def create_vocabulary(training_set):
	vocab = set()
	for ts in training_set:
		vocab.update(extract_bigrams_from_corpus(extract_words(ts[2])))
	return sorted(tuple(vocab))
"""
	[0] - count of labels of class in training data
	[1] - count of bigrams in [2]
	[2] - bigrams list
"""
def create_histograms(classes_labels):
	histograms = [{}, {}, {}]
	for line in classes_labels:
		histograms[0][line.strip()] = 0
		histograms[1][line.strip()] = 0
		# = {} for dictionary option
		histograms[2][line.strip()] = []
	return histograms

def create_bigrams(training_set, classes_labels):
	fail_counter = 0
	vocabulary = create_vocabulary(training_set)
	histograms = create_histograms(classes_labels)
	for lab in histograms[2]:
		histograms[2][lab] = np.zeros(len(vocabulary), dtype=int)
	for ts in training_set:
		f_bis = extract_bigrams_from_corpus(extract_words(ts[2]))
		for bi in f_bis:
			for lab in extract_annotations(ts[0]):
				try:
					histograms[2][lab][vocabulary.index(bi)] += f_bis[bi]
				except KeyError:
					fail_counter += 1
	print(f"Failed bigram additions: {fail_counter}")
	return [histograms, vocabulary]

"""
def create_bigrams(classes_lables, training_set):
	#vocab = set()
	vocab_len = 0
	histograms = create_histograms(classes_lables)
	for t in training_set:
		for an in extract_annotations(t[0]):
			#? TODO: testing for correct annotation?
			# TODO: 2) if file is only 1 word long?
			histograms[0][an] += 1
			words = extract_words(t[2])
			if len(words) == 0:
				continue
			histograms[2][an][words[0]] = 1
			#vocab.add(word[0])
			for w in range(1, len(words)):
				try:
					histograms[2][an][words[w - 1] + " " + words[w]] += 1
					#vocab.add(words[w - 1] + " " + words[w])
				except KeyError:
					histograms[2][an][words[w - 1] + " " + words[w]] = 1
					#vocab.add(words[w - 1] + " " + words[w])
				histograms[1][an] += 1
				vocab_len += 1
			histograms[2][an][words[len(words) - 1]] = 1
			#vocab.add(words[len(words) - 1])
			vocab_len += 2
			histograms[1][an] += 2
	return [histograms, vocab_len]

def extract_bigrams_from_corpus(parsed_file_data):
	bigrams = {}
	bigrams[parsed_file_data[0]] = 1
	for d in range(1, len(parsed_file_data)):
		try:
			bigrams[parsed_file_data[d - 1] + " " + parsed_file_data[d]] += 1
		except KeyError:
			bigrams[parsed_file_data[d - 1] + " " + parsed_file_data[d]] = 1
	bigrams[parsed_file_data[len(parsed_file_data) - 1]] = 1
	return bigrams
"""

if __name__ == "__main__":
	print('This is Bigram')
