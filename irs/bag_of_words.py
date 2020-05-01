import re
import glob
import numpy as np

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
		classes[line.strip()] = []		
	f.close()
	return classes

def create_classes_vectors(classes_file, files, vocabulary):
	classes = create_classes(classes_file)
	counter = 0
	for c in classes:
		classes[c] = np.zeros(len(vocabulary), dtype = int)
	for f in files:
		lines = open(f, "r", encoding='utf-8-sig').readlines()
		for an in extract_annotations(lines[0]):
			# SOLVED: find out why some an is 4 long and strip nor replace doesn't work
			# (0xff na prvnim indexu) --- NEED TO USE utf-8-sig ENCODING!!!
			#if len(an) == 4:
			#	an = an[1:]
			if an.strip() in classes:
				for w in extract_words(lines[2]):
					classes[an][vocabulary.index(w)] += 1
			else: 
				print(f"Unknown annotation: {an}")
				counter += 1
	print(f"fails: {counter}")
	return classes

if __name__ == "__main__":
	annotations = "fin pol arm poc"
	article = "Ahoj, jak se    dnes!!\" \n * vede?("
	#print(extract_annotations(annotations))
	#print(extract_words(article))
	files = glob.glob("../data/Train/*.lab")
	v = create_vocabulary(files)
	#print(len(v))

	#print(classes)
	classes = create_classes_vectors("../soubor_klas_trid.txt", files, v)
	print(classes)
