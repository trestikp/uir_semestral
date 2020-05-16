from math import log
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial


def classes_including_word_count(classes, w_index):
	count = 0
	for c in classes:
		if classes[c][2][w_index] > 0:
			count += 1
	return count

def compute_cc_vector(classes, vocab_len):
	pool = Pool(cpu_count())
	res = pool.map(partial(classes_including_word_count, classes), range(vocab_len))
	return res

def tfidf_one_class(classs, class_count, w_index, cc):
	#print(classs)
	new_class = [classs[0], classs[1], np.zeros(len(classs[2]))]
	tf = classs[2][w_index] / float(classs[1])
	idf = log(len(class_count) / (1 + float(cc)))
	new_class[c][2][w_index] = tf * idf
	return new_class



def bow_tfidf(classes, vocab_len):
	classes_new = {}
	for c in classes:
		classes_new[c] = [classes[c][0], classes[c][1], []]
		classes_new[c][2] = np.zeros(vocab_len, dtype=int)
	for w in range(vocab_len):
		cc = classes_including_word_count(classes, w)
		for c, val in classes.items():
			tf = val[2][w] / float(val[1])
			idf = log((len(classes) + 1) / (1 + float(cc)))
			classes_new[c][2][w] = round(tf * idf * classes_new[c][1])
			#classes_new[c][2][w] = round(tf * idf * vocab_len)
#	for c, val in classes_new.items():
#		for w in val[2]:
#			w = round(w * vocab_len)
		
	return classes_new

def bow_blabla(classes, vocab_len):
	pool = Pool(cpu_count())
	for w in range(vocab_len):
		cc = classes_including_word_count(classes, w)
		res = pool.map(partial(tfidf_one_class, class_count=len(classes), w_index=w, cc=cc), classes.values())
	return res


if __name__ == "__main__":
	None
"""
	documentA = 'the man went out for a walk'
	documentB = 'the children sat around the fire'
	#vocab = ["Tohle", "je", "prvni", "priklad", "druhy"]
	#str1 = ["Tohle", "je", "prvni", "priklad"]
	#str2 = ["Tohle", "je", "druhy", "priklad", "priklad"]
	str1 = documentA.split()
	str2 = documentB.split()
	vocab = sorted(list(set(str1) | set(str2)))
	classes = {"pr1": [1, 7, [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1]], "pr2": [1, 6, [0, 1, 1, 1, 0, 0, 0, 1, 2, 0, 0]]}

	print(classes)
	res = bow_tfidf(classes)
	print(res)
"""
