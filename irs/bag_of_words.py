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

def create_vocabulary(files):
	vocabulary = set()
	for f in files:
		lines = open(f, "r").readlines()
		vocabulary.update(extract_words(lines[2])) 
	return sorted(tuple(vocabulary)) #tuple for indexes, sorted to keep indexes consistent


def create_classes(class_file):
	classes = {}
	f = open(class_file, "r")
	for line in f.readlines():
		classes[line.strip()] = [0, 0, []]
	f.close()
	return classes

"""
def create_classes(class_file_buff):
	classes = {}
	for line in class_file_buff:
		classes[line.strip()] = [0, 0, []]
	return classes

def single_class_vector(classes, vocabulary, training_file):
	for an in training_file[0]:
		if an.strip() in classes:
			classes[an][0] += 1
			for w in training_file[1]:
				classes[an][2][vocabulary.index(w)] += 1
				classes[an][1] += 1
		else: 
			print(f"Unknown annotation: {an}")


def create_classes_vectors(training_files_set, classes, vocabulary):
	for c in classes:
		classes[c][2] = np.zeros(len(vocabulary), dtype = int)
	pool = Pool(cpu_count())
	pool.map(partial(single_class_vector, classes, vocabulary), training_files_set)
	return classes
"""

def create_classes_vectors(classes_file, files, vocabulary):
	classes = create_classes(classes_file)
	counter = 0
	for c in classes:
		#classes[c] = np.zeros(len(vocabulary), dtype=int)
		classes[c][2] = np.zeros(len(vocabulary), dtype = int)
	for f in files:
		lines = open(f, "r", encoding='utf-8-sig').readlines()
		for an in extract_annotations(lines[0]):
			# SOLVED: find out why some an is 4 long and strip nor replace doesn't work
			# (0xff na prvnim indexu) --- NEED TO USE utf-8-sig ENCODING!!!
			#if len(an) == 4:
			#	an = an[1:]
			if an.strip() in classes:
				#annot_count[an] += 1
				classes[an][0] += 1
				for w in extract_words(lines[2]):
					#classes[an][vocabulary.index(w)] += 1
					#annot_count[an] += 1
					classes[an][2][vocabulary.index(w)] += 1
					classes[an][1] += 1
			else: 
				print(f"Unknown annotation: {an}")
				counter += 1
	print(f"Unrecognized annotations: {counter}")
	return classes

if __name__ == "__main__":
	print('This is BoW')
