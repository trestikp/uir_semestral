from math import log
from multiprocessing import Pool, cpu_count
from functools import partial

def total_label_count(cl_histogram):
	res = 0
	for c in cl_histogram:
		res += cl_histogram[c]
	return res

def calculate_vocab_length(c_vectors):
	res = 0
	for c in c_vectors:
		res += len(c_vectors[c])
	return res

def compute_word_probability(vocab_len, word_cnt, class_wc):
	m = 1
	#m = 2
	p = 0.0000000000000001# - cca 23% acc
	#p = 0.000001 # cca 15.8 acc
	#p = 0.5

	#if word_cnt == 0:
	#	return 0
	numerator = word_cnt + (p * m)

	denominator = class_wc + (m * vocab_len)
	#denominator = class_wc + m

	return log(numerator / denominator)

def compute_class_probablity(class_cnt, total_usage):
	return log(class_cnt / total_usage)

def calculate_probabilities(parsed_file, c_labels, c_wcount, c_vectors):
	results = {}
	prob = 0
	missing_wc = 0
	total_lc = total_label_count(c_labels)
	vocab_len = calculate_vocab_length(c_vectors)

	for c in c_vectors:
		prob += compute_class_probablity(c_labels[c], total_lc)
		for word in parsed_file:
			try:
				#prob += parsed_file[word] * log(compute_word_probability(vocab_len, c_vectors[c][word], c_wcount[c]))
				prob += parsed_file[word] * compute_word_probability(vocab_len, c_vectors[c][word], c_wcount[c])
			except KeyError:
				#prob += parsed_file[word] * log(compute_word_probability(vocab_len, 0, c_wcount[c]))
				missing_wc += 1

		#prob += missing_wc * log(compute_word_probability(vocab_len, 0, c_wcount[c]))
		prob += missing_wc * compute_word_probability(vocab_len, 0, c_wcount[c])
		results[c] = prob
		prob = 0
		missing_wc = 0

	return results
	
def get_class(results):
	results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}
	#diff = results[max(results, key=results.get)] - results[min(results, key=results.get)]
	#podminka = results[min(results, key=results.get)] + (diff * 0.90)
	#for res in results:
	#	if float(results[res]) > podminka:
	#		new_res.append(res)
	#res = next(iter(results))
	#return list(results.keys())[len(results) - 1]
	return list(results.keys())[-1]

def classify_text_only(model, text):
	#presuming text is parsed
	prob = calculate_probabilities(text, model[0], model[1], model[2])
	classification = get_class(prob)
	return classification
	

def classify_file(model, file_data):
	prob = calculate_probabilities(file_data[2], model[0], model[1], model[2])
	classification = get_class(prob)
	accuracy = 0
	if classification in file_data[1]:
		accuracy = 1
	#accuracy = len(set(file_data[1]) & set(classification)) / len(set(classification) | set(file_data[1]))
	return [file_data[0], file_data[1], classification, accuracy]

def classify_file_set(model, test_set): 
	pool = Pool(int(cpu_count() * 3 / 4))
	class_res = pool.map(partial(classify_file, model), test_set)
	return class_res

def calculate_total_acc(classification_results):
	s = 0
	for cr in classification_results:
		s += cr[3]
	return (s / len(classification_results) * 100)
