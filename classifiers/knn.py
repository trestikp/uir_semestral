from math import sqrt

def calculate_distances(classes_vect, target_vect):
	distances = {}
	dist = 0
	for k, v in classes_vect.items():
		for x in range(len(target_vect)):
			dist += ((v[x] - target_vect[x]) * (v[x] - target_vect[x]))
		distances[k] = sqrt(dist)
		dist = 0
	return {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

def classify(file_data_set, classes_vect):
	distanes = []
	for fds in file_data_set:
		distanes.append([fds[0], calculate_distances(classes_vect, fds[2])])
	return distanes
