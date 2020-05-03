import math

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

def classify_file(parsed_file, vocabulary, classes):
	results = {}
	prob = 1
	for c in classes:
		prob += math.log(compute_class_probablity_s(classes[c], total_classes_usage(classes)))
		#prob *= compute_class_probablity_s(classes[c], total_classes_usage(classes))
		#print(prob)
		for word in parsed_file:
			prob += math.log(compute_word_probability_s(vocabulary, word, classes[c]))
			#prob *= compute_word_probability_s(vocabulary, word, classes[c])
		#	print(prob)
		#results.append([c, prob])
		results[c] = prob
		prob = 1
		#break;
	return results
	
def get_classes(results):
	new_res = []
	mi = min(results, key=results.get)
	ma = max(results, key=results.get)
	#diff = max(results) - min(results)
	diff = results[ma] - results[mi]
	podminka = results[mi] + (diff * 0.90)
	#print(f"min: {mi}	max: {ma}")
	#print(f"diff: {diff}")
	#print(f"podminka: {podminka}")
	for res in results:
		if float(results[res]) > podminka:
			new_res.append(res)
	return new_res
