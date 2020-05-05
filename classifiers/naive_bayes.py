import math
from multiprocessing import Pool, cpu_count
from functools import partial

def total_classes_usage(classes):
	res = 0
	for c in classes:
		res += classes[c][0]
	return res

def compute_word_probability(vocabulary, classes, word, classs):
	numerator  = classes[classs][2][vocabulary.index(word)] + 1
	denominator = classes[classs][1] + len(vocabulary)
	return numerator / denominator

def compute_word_probability_s(vocabulary, word, classs):
	try:
		numerator  = classs[2][vocabulary.index(word)] + 1
	except ValueError:
		numerator = 1
	denominator = classs[1] + len(vocabulary)
	return numerator / denominator

def compute_class_probablity(classes, classs, file_count):
	return classes[classs][0] / file_count	

def compute_class_probablity_s(classs, total_usage):
	return classs[0] / total_usage

def ccp(class_wc, voc_len):
	return class_wc / voc_len

def calculate_probabilities(parsed_file, vocabulary, classes):
	results = {}
	prob = 0
	for c in classes:
		#prob += math.log(compute_class_probablity_s(classes[c], total_classes_usage(classes)))
		prob += math.log(ccp(classes[c][1], len(vocabulary)))
		for word in parsed_file:
			prob += math.log(compute_word_probability_s(vocabulary, word, classes[c]))
		results[c] = prob
		prob = 0
	return results
	
def get_classes(results):
	new_res = []
	diff = results[max(results, key=results.get)] - results[min(results, key=results.get)]
	podminka = results[min(results, key=results.get)] + (diff * 0.90)
	for res in results:
		if float(results[res]) > podminka:
			new_res.append(res)
	return new_res

def classify_file(vocabulary, classes, file_data):
	probs = calculate_probabilities(file_data[2], vocabulary, classes)
	classification = get_classes(probs)
	#accuracy = 1 - len(set(file_data[1]) | set(classification)) / (len(classification) + len(file_data[1]))
	#accuracy = len(set(file_data[1]) & set(classification)) / len(set(classification) | set(file_data[1]))
	accuracy = len(set(file_data[1]) & set(classification)) / len(classification)
	#if file_data[0] == "data/Test/posel-od-cerchova-1873-01-11-n2_0080_4.lab":
	#	print(f"{len(set(file_data[1]) & set(classification))}/{(len(classification) | len(file_data[1]))}")
	return [file_data[0], file_data[1], classification, accuracy]

def classify_file_set(file_data_set, vocabulary, classes): 
	pool = Pool(cpu_count())
	class_res = pool.map(partial(classify_file, vocabulary, classes), file_data_set)
	return class_res

def calculate_total_acc(classification_results):
	s = 0
	for cr in classification_results:
		s += cr[3]
	return s / len(classification_results)
