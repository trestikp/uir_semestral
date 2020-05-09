import glob
import time

import irs.bag_of_words as bow
import irs.bag_of_words2 as bow2
import classifiers.naive_bayes as nb
import classifiers.naive_bayes2 as nb2
import irs.tfidf as tfidf
import classifiers.knn as knn
import irs.bigram as bigram


def create_model(training_files, class_label_file, method=0):
	training_set = []
	for f in training_files:
		temp = open(f, "r", encoding="utf-8-sig")
		training_set.append(temp.readlines())
		temp.close()
	cl_file = open(class_label_file, "r", encoding="utf-8-sig")
	cl_data = cl_file.readlines()
	cl_file.close()

	output = None
	if method == 0:
		output = bow2.create_bow_different(training_set, cl_data)
	elif method == 1:
		output = bigram.create

	return output

def nb_class(vocabulary, model, test_files):
	test_set = []
	for f in test_files:
		temp = open(f, "r", encoding="utf-8-sig")
		lines = temp.readlines()
		test_set.append([temp.name, bow2.extract_annotations(lines[0]), bow.extract_words(lines[2])])
		temp.close()

	results = nb2.classify_file_set(vocabulary, model, test_set)
	return results
	

if __name__ == "__main__":
	mod_voc = create_model(glob.glob("./data/Train/*.lab"), "./soubor_klas_trid.txt", 0)
	clas_results = nb_class(mod_voc[1], mod_voc[1], glob.glob("./data/Test/*.lab"))
	print(f"Total accuracy: {nb2.calculate_total_acc(clas_results)}")
