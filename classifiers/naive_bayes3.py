from math import log
from multiprocessing import Pool, cpu_count
from functools import partial

def total_label_count(cl_histogram):
	res = 0
	for c in cl_histogram:
		#print(f"here?{c}")
		res += cl_histogram[c]
	return res

def compute_word_probability(vocabulary, word_cnt, class_wc):
	#m = 1
	m = 2
	#p = 0.0000000000000001 - cca 19.4 acc
	p = 0.000001 # cca 15.8 acc
	#p = 0.5
	try:
		numerator = word_cnt + (p * m)
	except ValueError:
		numerator = (p * m)
		#numerator = 0
	denominator = class_wc + (m * len(vocabulary))
	#denominator = class_wc + m
	return numerator / denominator

def compute_class_probablity(class_cnt, total_usage):
	return class_cnt / total_usage

def calculate_probabilities(parsed_file, vocabulary, c_labels, c_wcount, c_vectors):
	results = {}
	prob = 0
	total_lc = total_label_count(c_labels)
	for c in c_vectors:
		class_prob = compute_class_probablity(c_wcount[c], total_lc)
		#class_prob = c_wcount[c] / len(vocabulary)
		prob += log(class_prob)
		for word in parsed_file:
			try:
				prob += parsed_file[word] * log(compute_word_probability(vocabulary, c_vectors[c][vocabulary.index(word)], c_wcount[c]))
			except ValueError:
				prob += parsed_file[word] * log(compute_word_probability(vocabulary, 0, c_wcount[c]))
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

def classify_file(vocabulary, model, file_data):
	probs = calculate_probabilities(file_data[2], vocabulary, model[0], model[1], model[2])
	classification = get_classes(probs)
	#accuracy = 1 - len(set(file_data[1]) | set(classification)) / (len(classification) + len(file_data[1]))
	accuracy = len(set(file_data[1]) & set(classification)) / len(set(classification) | set(file_data[1]))
	#accuracy = len(set(file_data[1]) & set(classification)) / len(classification)
	if file_data[0] == "./data/Test/posel-od-cerchova-1873-01-11-n2_0080_4.lab":
		probs = {k: v for k, v in sorted(probs.items(), key=lambda item: item[1])}
		for p in probs:
			print(f"{p} - {probs[p]}")
		print(f"{len(set(file_data[1]) & set(classification))}/{(len(classification) | len(file_data[1]))}")
	return [file_data[0], file_data[1], classification, accuracy]

def classify_file_set(vocabulary, model, test_set): 
	pool = Pool(int((cpu_count() * 3) / 4))
	class_res = pool.map(partial(classify_file, vocabulary, model), test_set)
	return class_res

def calculate_total_acc(classification_results):
	s = 0
	for cr in classification_results:
		s += cr[3]
	return (s / len(classification_results) * 100)
