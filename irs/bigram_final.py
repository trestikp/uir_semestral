from sys import path
path.append("..")
import utility

def extract_bigrams_from_file(parsed_file_data):
	bigrams = {}
	if len(parsed_file_data) > 0:
		bigrams[parsed_file_data[0]] = 1
		for d in range(1, len(parsed_file_data)):
			try:
				bigrams[parsed_file_data[d - 1] + " " + parsed_file_data[d]] += 1
			except KeyError:
				bigrams[parsed_file_data[d - 1] + " " + parsed_file_data[d]] = 1
		bigrams[parsed_file_data[len(parsed_file_data) - 1]] = 1
	return bigrams

"""
	[0] - count of labels of class in training data
	[1] - count of bigrams in [2]
	[2] - bigrams list
"""
def create_histograms(classes_labels):
	histograms = [{}, {}, {}]
	for line in classes_labels:
		histograms[0][line.strip()] = 0
		histograms[1][line.strip()] = 0
		# = {} for dictionary option
		histograms[2][line.strip()] = []
	return histograms

def fill_histograms(training_set, classes_labels):
	fail_counter = 0
	histograms = create_histograms(classes_labels)
	for ts in training_set:
		for lab in utility.extract_annotations(ts[0]):
			try:
				histograms[0][lab] += 1
				f_bis = extract_bigrams_from_file(utility.extract_words(ts[2]))
				histograms[1][lab] += len(f_bis)
				histograms[2][lab].append(f_bis)
			except KeyError:
				fail_counter += 1
	print(f"Failed bigram additions: {fail_counter}")
	return histograms

def unite_files(histograms):
	dictionary = {}
	for c, val in histograms[2].items():
		for fd in val:
			for el, val in fd.items():
				try:
					dictionary[el] += val
				except KeyError:
					dictionary[el] = val
		histograms[2][c] = dictionary
		dictionary = {}
	return histograms

def create_bigrams_files(training_set, classes_labels):
	histograms = fill_histograms(training_set, classes_labels)
	return histograms

def create_bigrams_vectors(training_set, classes_labels):
	histograms = unite_files(fill_histograms(training_set, classes_labels))
	return histograms


if __name__ == "__main__":
	print('This is Bigram')
