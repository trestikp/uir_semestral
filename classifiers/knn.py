from math import sqrt
from numpy import zeros

from multiprocessing import Pool, cpu_count
from functools import partial

def calculate_distances(c_vectors, target_vect):
	distances = {}
	dist = 0
	for k, v in c_vectors.items():
		for x in range(len(target_vect)):
			dist += ((v[x] - target_vect[x]) * (v[x] - target_vect[x]))
		distances[k] = sqrt(dist)
		dist = 0
	return {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

def vectorize_text(parsed_file, vocabulary):
	vect = []
	vect = zeros(len(vocabulary), dtype=int)
	for w in parsed_file:
		try:
			vect[vocabulary.index(w)] += 1
		except ValueError:
			continue
	return vect

def choose_classes(distances):
	results = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
	#result = []
	#diff = distances[max(distances, key=distances.get)] - distances[min(distances, key=distances.get)]
	#cond = distances[min(distances, key=distances.get)] + (diff * 0.7)
	#for k in distances:
		#print(f"cond: {cond} : {k} - {distances[k]}")
	#	if distances[k] > cond:
	#		result.append(k)
	#return result	
	return list(results.keys())[len(results) - 1]

def classify_file(vocabulary, c_vectors, file_data):
	target = vectorize_text(file_data[2], vocabulary)
	distances = calculate_distances(c_vectors, target)
	classification = choose_classes(distances)
	accuracy = 0
	if classification in file_data[1]:
		accuracy = 1
	#results = choose_classes(distances)
	#accuracy = len(set(file_data[1]) & set(results)) / len(set(results) | set(file_data[1]))
	if file_data[0] == './data/Test/posel-od-cerchova-1874-02-14-n7_0175_0.lab':
		for d in distances:
			print(f"{d} - {distances[d]}")
	return [file_data[0], file_data[1], classification, accuracy]
		
def classify_file_set(vocabulary, c_vectors, file_data_set):
	pool = Pool(int(cpu_count() * 3 / 4))
	class_res = pool.map(partial(classify_file, vocabulary, c_vectors), file_data_set)
	#class_res = classify_file(vocabulary, c_vectors, file_data_set[0])
	return class_res

def calculate_total_acc(class_res):
	s = 0
	for cr in class_res:
		s += cr[3]
	return (s / len(class_res) * 100)

if __name__ == "__main__":
	None
