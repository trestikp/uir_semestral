from math import log

def compute_idfs(c_dict_f, file_count):
	vocab = set()
	# labels
	for c in c_dict_f:
		#dict list
		for d in c_dict_f[c]:
			#print(d)
			#print(c_dict_f[c])
			vocab |= set(list(d.keys()))
	vocab = dict.fromkeys(vocab, 0)
	for c in c_dict_f:
		for d in c_dict_f[c]:
			for w in d:
				try:
					vocab[w] += 1
				except KeyError:
					print(f"Word {w} is not in vocabulary. Skipping word...")

	for w in vocab:
		try:
			#print(f"{file_count} / {vocab[w]}")
			vocab[w] = log(file_count) / (float(vocab[w]))
		except ZeroDivisionError:
			#this is usually prevented by adding +1 to numerator
			vocab[w] = 0
	return vocab
	

def compute_tf(word_count, class_wc):
	return word_count / float(class_wc)

def calculate_tf_idf(model, file_count):
	idfs = compute_idfs(model[2], file_count)
	for c in model[2]:
		for d in model[2][c]:
			for w in d:
				tf = compute_tf(d[w], model[1][c])
				d[w] = tf * idfs[w]
	return model

def total_number_of_words(c_wcount):
	total = 0
	for c in c_wcount:
		total += c_wcount[c]
	return total

def normalize_tf_idf(model):
	total = total_number_of_words(model[1])
	for c in model[2]:
		for d in model[2][c]:
			for w in d:
				d[w] = d[w] * total
	return model

if __name__ == "__main__":
	#documentA = 'the man went out for a walk'
	#documentB = 'the children sat around the fire'

	#one = "tohe je tf idf pokus"
	#two = "timto je tf idf hokus"
	fake_model = [{"one": 1,"two": 1},\
		      {"one": 7,"two": 6},\
		      {"one": {"the": 1, "man": 1, "went": 1, "out": 1,\
		      	       "for": 1, "a": 1, "walk": 1},\
		       "two": {"the": 2, "children": 1, "sat": 1, "around": 1,\
		       	       "fire": 1}}]
	print(f"before tfidf: {fake_model}\n")
	calculate_tf_idf(fake_model)
	print(f"after tfidf: {fake_model}\n")
	normalize_tf_idf(fake_model)
	print(f"normalized tfidf: {fake_model}\n")
