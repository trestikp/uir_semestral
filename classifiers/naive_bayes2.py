import math
import time
from multiprocessing import Pool, cpu_count
from functools import partial

def total_classes_usage(cls_histogram):
	res = 0
	for c, v in cls_histogram.items():
		res += v
	return res

"""
def compute_word_probability(vocabulary, word, classs, classs_wc):
	m = 1
	p = 0.1
	try:
		numerator  = classs[vocabulary.index(word)] + (p * m)
	except ValueError:
		numerator = (p * m)
	denominator = classs_wc + (m * len(vocabulary))
	return numerator / denominator
"""
def compute_word_probability(vocabulary, word_cnt, classs_wc):
	m = 1
	p = 0.1
	if word_cnt > 0:
		numerator  = word_cnt + (p * m)
	else:
		numerator = (p * m)
	denominator = classs_wc + (m * len(vocabulary))
	#denominator = classs_wc + (m)
	return numerator / denominator

def compute_class_probablity(classs_ac, total_usage):
	return classs_ac / total_usage

def train_classifier(vocabulary, cls_histogram, vec_histogram, wc_histogram):
	prob_table = {}
	prob_classes = {}
	miscellaneous = {}
	total_cls_usage = total_classes_usage(cls_histogram)
	for ch, ch_v in cls_histogram.items():
		prob_classes[ch] = compute_class_probablity(ch_v, total_cls_usage)
		#prob_classes[ch] = compute_class_probablity(ch_v, 424)
	for vh, vh_v in vec_histogram.items():
		prob_table[vh] = []
		for word in vh_v:
			prob_table[vh].append(compute_word_probability(vocabulary, word, wc_histogram[vh]))
	miscellaneous["not_in_class"] = {}
	for vh, vh_v in vec_histogram.items():
		miscellaneous["not_in_class"][vh] = compute_word_probability(vocabulary, 0, wc_histogram[vh])
	
	return [prob_table, prob_classes, miscellaneous]
		
def calculate_probabilities(vocabulary, parsed_file, prob_table, prob_classes, prob_misc):
	results = {}
	prob = 0
	for pc, pc_v in prob_classes.items():
		prob += math.log(pc_v)
		for w in parsed_file:
			try:
				prob += math.log(prob_table[pc][vocabulary.index(w)])
			except ValueError:
				prob += math.log(prob_misc["not_in_class"][pc])
		results[pc] = prob
		prob = 0
	return results

def extract_classes(prob_results):
	results = []
	diff = prob_results[max(prob_results, key=prob_results.get)] - prob_results[min(prob_results, key=prob_results.get)]
	podminka = prob_results[min(prob_results, key=prob_results.get)] + (diff * 0.90)
	for res in prob_results:
		if float(prob_results[res]) > podminka:
			results.append(res)
	return results


def classify_file(vocabulary, model, file_data):
	start = time.time()
	probs = calculate_probabilities(vocabulary, file_data[2], model[0], model[1], model[2])
	end = time.time()
	#print(f"Calculating probabilities took: {end - start}")
	classification = extract_classes(probs)
	accuracy = len(set(file_data[1]) & set(classification)) / len(set(classification) | set(file_data[1]))
	#accuracy = len(set(file_data[1]) & set(classification)) / len(classification)
	return [file_data[0], file_data[1], classification, accuracy]

def classify_file_set(vocabulary, model, test_set):
	pool = Pool(cpu_count())
	results = pool.map(partial(classify_file, vocabulary, model), test_set)
	#for fds in file_data_set:
	#	results.append(classify_file(vocabulary, fds, model))
	return results

def calculate_total_acc(classification_results):
	s = 0
	for cr in classification_results:
		s += cr[3]
	return s / len(classification_results)

"""
def calculate_probabilities(parsed_file, vocabulary, classes):
	results = {}
	prob = 0
	for c in classes:
		prob += math.log(compute_class_probablity_s(classes[c], total_classes_usage(classes)))
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
	accuracy = len(set(file_data[1]) & set(classification)) / len(set(classification) | set(file_data[1]))
	#accuracy = len(set(file_data[1]) & set(classification)) / len(classification)
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
"""
