import re
import glob
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial


def extract_words(article):
	article = re.sub("[^\w]", " ", article).split()
	return [word.lower().strip() for word in article]

def extract_annotations(line):
	line = line.split()
	return [l.strip() for l in line]

def create_vocabulary(training_set):
	vocabulary = set()
	for ts in training_set:
		vocabulary.update(extract_words(ts[2])) 
	return sorted(tuple(vocabulary)) #tuple for indexes, sorted to keep indexes consistent


def create_histograms(classes_file_content):
	histograms = [{}, {}, {}]
	for line in classes_file_content:
		histograms[0][line.strip()] = 0
		histograms[1][line.strip()] = []
		histograms[2][line.strip()] = 0
	return histograms

def fill_classes_histograms(classes_file_content, training_set, vocabulary):
	histograms = create_histograms(classes_file_content)
	counter = 0
	for k in histograms[1]:
		histograms[1][k] = np.zeros(len(vocabulary), dtype = int)
	for ts in training_set:
		for an in extract_annotations(ts[0]):
			# SOLVED: find out why some an is 4 long and strip nor replace doesn't work
			# (0xff na prvnim indexu) --- NEED TO USE utf-8-sig ENCODING!!!
			#if len(an) == 4:
			#	an = an[1:]
			if an.strip() in histograms[0]:
				histograms[0][an] += 1
				for w in extract_words(ts[2]):
					#print(histograms[1][an])
					histograms[1][an][vocabulary.index(w)] += 1
					histograms[2][an] += 1
			else: 
				print(f"Unknown annotation: {an}")
				counter += 1
	print(f"Unrecognized annotations: {counter}")
	return histograms

def create_bow(training_files, class_file):
	training_set = []
	for ts in training_files:
		f = open(ts, "r", encoding="utf-8-sig")
		training_set.append(f.readlines())
		f.close()

	f = open(class_file, "r", encoding="utf-8-sig")
	clc = f.readlines()
	f.close()

	vocabulary = create_vocabulary(training_set)
	histograms = fill_classes_histograms(clc, training_set, vocabulary)

	return [histograms, vocabulary]

if __name__ == "__main__":
	print('This is BoW')
