import re
import glob
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import partial

from sys import path
path.append("..")
import utility

"""
	Initializes histograms for classes. 
	histogram[0] = classes annotations count across files
	histogram[2] = classes tf vector
	histogram[1] = classes word count
"""
def create_histograms(classes_file_content):
	histograms = [{}, {}, {}]
	for line in classes_file_content:
		histograms[0][line.strip()] = 0
		histograms[1][line.strip()] = 0
		histograms[2][line.strip()] = []
	return histograms

def file_to_dictionary(text):
	result = {}
	#stop-words?
	for w in utility.extract_words(text):
		try:
			result[w] += 1
		except KeyError:
			result[w] = 1
	return result

def fill_classes_histograms(training_set, classes_file_content):
	histograms = create_histograms(classes_file_content)
	counter = 0
	#for c in histograms[2]:
	#	for k in histograms[2]
	#		histograms[2][c][k] = []
	for ts in training_set:
		for an in utility.extract_annotations(ts[0]):
			if an.strip() in histograms[0]:
				histograms[0][an] += 1
				#print(file_to_dictionary(ts[2]))
				histograms[2][an].append(file_to_dictionary(ts[2]))
				histograms[1][an] += len(histograms[2][an][-1])
			else: 
				counter += 1
	print(f"Unrecognized annotations: {counter}")
	return histograms

def unite_files(histograms):
	dictionary = {}
	#c val fd   el
	#{ [   {}, {asdf: x}, {}... ] }
	for c, val in histograms[2].items():
		for fd in val:
			for el, val in fd.items():
				try: 
					dictionary[el] += val
				except KeyError:
					dictionary[el] = val
		histograms[2][c] = dictionary
		dictionary = {}
	return histograms

def create_bows_files(training_set, class_labels):
	histograms = fill_classes_histograms(training_set, class_labels)
	return histograms

def create_bows_vectors(training_set, class_labels):
	histograms = unite_files(fill_classes_histograms(training_set, class_labels))
	return histograms

if __name__ == "__main__":
	print('This is BoW')
