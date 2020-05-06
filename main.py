import glob

import irs.bag_of_words as bow
import irs.bag_of_words2 as bow2
import classifiers.naive_bayes as nb
import irs.tfidf as tfidf

"""
def bow_create():
	tf_set = []
	training_files = glob.glob("data/Train/*.lab")
	for f in training_files:
		tf = open(f, "r", encoding="utf-8-sig")
		lines = tf.readlines()
		tf_set.append([bow.extract_annotations(lines[0]), bow.extract_words(lines[2])])
		tf.close()
	c_file = open("./soubor_klas_trid.txt", "r")
	classes = bow.create_classes(c_file.readlines())
	c_file.close()
	#print(tf_set[1][0])
	#print(tf_set[1][1])
	return bow.create_classes_vectors(tf_set, classes, vocab)
"""

def nb_class():	
	file_data_set = []
	test_files = glob.glob("data/Test/*.lab")
	for f in test_files:
		tf = open(f, "r", encoding="utf-8-sig")
		lines = tf.readlines()
		file_data_set.append([tf.name, bow.extract_annotations(lines[0]), bow.extract_words(lines[2])])
		tf.close()
	return nb.classify_file_set(file_data_set, vocab, classes)

if __name__ == "__main__":
#	vocab = bow.create_vocabulary(glob.glob("data/Train/*.lab"))
	#vocab2 = bow.create_vocabulary(glob.glob("data/Test/*.lab"))
	#vocab += vocab2
#	classes = bow.create_classes_vectors("./soubor_klas_trid.txt", glob.glob("data/Train/*.lab"), vocab)
	#classes = bow_create()
	#for c in classes:
	#	print(f"{c}  -   {classes[c]}") 

	bow_stuff = bow2.create_bow(glob.glob("data/Train/*.lab"), "./soubor_klas_trid.txt")
	for bs in bow_stuff[0]:
		for k, v in bs.items():
			print(f"{k}  -  {v}")
	
	print(f"vocab len: {len(bow_stuff[1])}")
	
	#classes = tfidf.bow_tfidf(classes, len(vocab))
	#print(tfidf.bow_tfidf(classes, len(vocab)))
	#for c in cls:
	#	print(f"{c}  -   {cls[c]}") 

	#for f, u in cls.items():
	#	for c in u[2]:
	#		if c < 0:
	#			print(c)
"""
	nb_res = nb_class()
	for n in nb_res:
		#if n[0] == "data/Test/posel-od-cerchova-1873-01-11-n2_0080_4.lab":
		#	for i in range(len(n[1])):
		#		print(" ".join(hex(ord(n)) for n in n[1][i]))
		#	for i in range(len(n[2])):
		#		print(" ".join(hex(ord(n)) for n in n[2][i]))
		print(n)

	print(f"total acc: {nb.calculate_total_acc(nb_res)}")
"""
