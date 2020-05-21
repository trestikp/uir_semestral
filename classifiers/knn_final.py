from math import sqrt
from numpy import zeros

from multiprocessing import Pool, cpu_count
from functools import partial

def calculate_distances(c_vectors_f, target_dict):
	distances = {}

	for f in c_vectors_f:
		dist_list = []
		for d in c_vectors_f[f]:
			dist = 0
			vocab = d.keys() | target_dict.keys()
			for w in vocab:
				try:
					x = d[w]
				except KeyError:
					x = 0
				try:
					y = target_dict[w]
				except KeyError:
					y = 0
				dist += ((x - y) * (x - y))

			dist_list.append(sqrt(dist))
		distances[f] = sorted(dist_list)

	return distances

def dict_file(parsed_file):
	dictionary = {}

	for w in parsed_file:
		try:
			dictionary[w] += 1
		except KeyError:
			dictionary[w] = 1

	return dictionary

def choose_class(result):
	first = 0
	second = 0
	freq = 0
	res = None

	for lab in result:
		freq = result.count(lab)
		if freq > first:
			second = first
			first = freq
			res = lab

	if second == first:
		return True
	return res


		

def choose_classification(distances):
	classification = None
	result = []
	rounds = 0
	round_limit = 13
	
	while len(result) < 5 and choose_class(result) and rounds < round_limit:
		key = None
		val = 10000000.0
		for d, v in distances.items():
			try:
				if v[rounds] < val:
					val = v[rounds]
					key = d
			except IndexError:
				continue

		result.append(key)
		rounds += 1
	
	classification = choose_class(result)

	if classification == True:
		print("Could not determine class")
		print(result)
		return None

	return classification

def classify_file(c_vectors_f, file_data):
	target = dict_file(file_data[2])
	distances = calculate_distances(c_vectors_f,target)
	classification = choose_classification(distances)
	accuracy = 0

	if classification in file_data[1]:
		accuracy = 1

	return [file_data[0], file_data[1], classification, accuracy]

def classify_text_only(c_vectors_f, text):
	#text is parsed beforehand
	target = dict_file(text)
	distances = calculate_distances(c_vectors_f,target)
	classification = choose_classification(distances)
	return classification

		
def classify_file_set(c_vectors_f, file_data_set):
	pool = Pool(int(cpu_count() * 3 / 4))
	class_res = pool.map(partial(classify_file, c_vectors_f), file_data_set)
	return class_res

def calculate_total_acc(class_res):
	s = 0
	for cr in class_res:
		s += cr[3]
	return (s / len(class_res) * 100)

if __name__ == "__main__":
	None
