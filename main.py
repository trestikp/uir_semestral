import glob
import time
from re import sub

import utility

import irs.bag_of_words_vect as bow_v
import irs.bag_of_words_dict as bow_d
import classifiers.naive_bayes_vect as nb_v
import classifiers.naive_bayes_dict as nb_d
import irs.tfidf_dict as tfidf_d
import irs.tfidf_vect as tfidf_v
import classifiers.knn as knn
import irs.bigram_vect as bigram_v
import irs.bigram_dict as bigram_d


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
		output = bow_v.create_bow_different(training_set, cl_data)
	elif method == 1:
		output = bigram_v.create_bigrams(training_set, cl_data)
	elif method == 2:
		output = bow_d.create_bow(training_set, cl_data)
	elif method == 3:
		output = bigram_d.create_bigrams(training_set, cl_data)
	elif method == 4:
		m_a_v = bow_d.create_bow(training_set, cl_data)
		#output = [tfidf_d.calculate_tf_idf(m_a_v[0]), m_a_v[1]]
		output = [tfidf_d.normalize_tf_idf(tfidf_d.calculate_tf_idf(m_a_v[0])), m_a_v[1]]
	elif method == 5:
		m_a_v = bigram_d.create_bigrams(training_set, cl_data)
		#output = [tfidf_d.calculate_tf_idf(m_a_v[0]), m_a_v[1]]
		output = [tfidf_d.normalize_tf_idf(tfidf_d.calculate_tf_idf(m_a_v[0])), m_a_v[1]]


	return output

def extract_words_with_count(text):
	text = utility.extract_words(text)
	#text = sub("[^\w]", " ", text).split()
	result = {}
	for w in text:
		# kupodivu funguje lip bez nasledujici radky, aspon s nb + bow
		# lepsi pro nb + bigram
		#w = w.lower()
		try:
			result[w] += 1
		except KeyError:
			result[w] = 1
	return result

def prepare_test_set(test_files):
	test_set = []
	for f in test_files:
		temp = open(f, "r", encoding="utf-8-sig")
		lines = temp.readlines()
		#test_set.append([temp.name, bow_v.extract_annotations(lines[0]), bow.extract_words(lines[2])])
		test_set.append([temp.name, utility.extract_annotations(lines[0]), extract_words_with_count(lines[2])])
		temp.close()
	#print(test_set[0])
	return test_set

def nb_v_class(vocabulary, model, test_files):
	test_set = prepare_test_set(test_files)
	#print(test_set)
	results = nb_v.classify_file_set(vocabulary, model, test_set)
	return results

def nb_d_class(model, test_files):
	test_set = prepare_test_set(test_files)
	results = nb_d.classify_file_set(model, test_set)
	return results

def knn_class(vocabulary, model, test_files):
	test_set = prepare_test_set(test_files)
	results = knn.classify_file_set(vocabulary, model[2], test_set)	
	return results
	
def classify(vocabulary, model, test_files, classifier=0):
	c_result = None
	if classifier == 0:
		c_result = nb_v_class(vocabulary, model, test_files)
	elif classifier == 1:
		c_result = nb_d_class(model, test_files)
	elif classifier == 2:
		c_result = knn_class(vocabulary, model, test_files)
	return c_result

	
def print_results(clas_results):
	for res in clas_results:
		print(res)

if __name__ == "__main__":
	model_and_vocabulary = create_model(glob.glob("./data/Train/*.lab"), "./soubor_klas_trid.txt", 4)
	c_results = classify(model_and_vocabulary[1], model_and_vocabulary[0], glob.glob("./data/Test/*.lab"), 1)
	print_results(c_results)
	print(f"Total accuracy: {nb_v.calculate_total_acc(c_results):.2f}%")
	#print(f"Total accuracy: {knn.calculate_total_acc(c_results):.2f}%")
