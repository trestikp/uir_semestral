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
		histograms[2][line.strip()] = {}
	return histograms

def fill_classes_histograms(training_set, classes_file_content):
	histograms = create_histograms(classes_file_content)
	counter = 0
	for ts in training_set:
		for an in utility.extract_annotations(ts[0]):
			if an.strip() in histograms[0]:
				histograms[0][an] += 1
				for w in utility.extract_words(ts[2]):
					#print(histograms[1][an])
					try:
						histograms[2][an][w] += 1
					except KeyError:
						histograms[2][an][w] = 1
					histograms[1][an] += 1
			else: 
				print(f"Unknown annotation: {an}")
				counter += 1
	print(f"Unrecognized annotations: {counter}")
	return histograms

def create_bow(training_set, class_labels):
	histograms = fill_classes_histograms(training_set, class_labels)
	return [histograms, None]

if __name__ == "__main__":
	print('This is BoW')
