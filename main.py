import glob
from multiprocessing import Pool

import irs.bag_of_words as bow
import classifiers.naive_bayes as nb

def some_f():
	overall_acc = 0
	test_files = glob.glob("data/Test/*.lab")
	for f in test_files:
		test_file = open(f, "r")
		lines = test_file.readlines()
		art = bow.extract_words(lines[2])
		res = nb.classify_file(art, vocab, classes)
		blah = nb.get_classes(res)
		annotations = bow.extract_annotations(lines[0])
		curr_acc = 0
		for annot in annotations:
			if annot not in blah:
				curr_acc += 1
		for bla in blah:
			if bla not in annotations:
				curr_acc += 1
		res_acc = curr_acc / (len(annotations) + len(blah))
		res_acc = 1 - res_acc
		print(f"Is: {annotations}  --  Classified: {blah}  --  Acc: {res_acc}")
		overall_acc += res_acc
		test_file.close()
	overall_acc = overall_acc / len(test_files)
	print(f"Total accuracy: {overall_acc}")


if __name__ == "__main__":
	#naive_bayes.train(glob.glob("data/Train/*.lab"), "./soubor_klas_trid.txt")
	vocab = bow.create_vocabulary(glob.glob("data/Train/*.lab"))
	classes = bow.create_classes_vectors("./soubor_klas_trid.txt", glob.glob("data/Train/*.lab"), vocab)
	for c in classes:
		print(f"{c}  -   {classes[c]}") 

	#p = Pool(processes=8)
	#p.apply_async(some_f)
	some_f()


	"""
	overall_acc = 0
	test_files = glob.glob("data/Test/*.lab")
	for f in test_files:
		test_file = open(f, "r")
		lines = test_file.readlines()
		art = bow.extract_words(lines[2])
		res = nb.classify_file(art, vocab, classes)
		blah = nb.get_classes(res)
		annotations = bow.extract_annotations(lines[0])
		curr_acc = 0
		for annot in annotations:
			if annot not in blah:
				curr_acc += 1
		for bla in blah:
			if bla not in annotations:
				curr_acc += 1
		res_acc = curr_acc / (len(annotations) + len(blah))
		res_acc = 1 - res_acc
		print(f"Is: {annotations}  --  Classified: {blah}  --  Acc: {res_acc}")
		overall_acc += res_acc
		test_file.close()
	overall_acc = overall_acc / len(test_files)
	print(f"Total accuracy: {overall_acc}")
	"""


	#art = bow.extract_words(test_file.readlines()[2])

	"""
	res = nb.classify_file(art, vocab, classes)
	print(res)
	blah = nb.get_classes(res)
	print(blah)
	"""
#	for f in res:
		#print("{:.5f}".format(f))
	#print("{:.10f}".format(res))
