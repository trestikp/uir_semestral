import argparse
import glob
from re import sub

import json

from tkinter import *

import utility

import classifiers.naive_bayes_dict as nb_d
import irs.tfidf_dict as tfidf_d
import irs.tfidf_files as tfidf_f
import classifiers.knn_final as knn_f

import irs.bag_of_words_final as bow_final
import irs.bigram_final as bi_f


def create_model(training_files, class_label_file, method=0, classifier=0):
	training_set = []
	for f in training_files:
		temp = open(f, "r", encoding="utf-8-sig")
		training_set.append(temp.readlines())
		temp.close()
	cl_file = open(class_label_file, "r", encoding="utf-8-sig")
	cl_data = cl_file.readlines()
	cl_file.close()

	output = None
	if method == 0:
		if classifier == 0:
			output = bow_final.create_bows_vectors(training_set, cl_data)
		elif classifier == 1:
			output = bow_final.create_bows_files(training_set, cl_data)
	elif method == 1:
		if classifier == 0:
			output = bi_f.create_bigrams_vectors(training_set, cl_data)
		elif classifier == 1:
			output = bi_f.create_bigrams_files(training_set, cl_data)
	elif method == 2:
		if classifier == 0:
			mav = bow_final.create_bows_vectors(training_set, cl_data)
			##output = tfidf_d.normalize_tf_idf(tfidf_d.calculate_tf_idf(mav))
			#output = tfidf_d.calculate_tf_idf(mav)
			output = tfidf_d.calculate_normalized_tf_idf(mav)
		elif classifier == 1:
			mav = bow_final.create_bows_files(training_set, cl_data)
			##output = tfidf_f.normalize_tf_idf(tfidf_f.calculate_tf_idf(mav, len(training_set)))
			#output = tfidf_f.calculate_tf_idf(mav, len(training_set))
			output = tfidf_f.calculate_normalized_tf_idf(mav, len(training_set))

	return output

def save_model(model, file_name, classifier):
	dump = json.dumps([classifier, model])
	f = open(file_name, "w", encoding="utf-8-sig")
	f.writelines(dump)

def extract_words_with_count(text):
	text = utility.extract_words(text)
	result = {}
	for w in text:
		try:
			result[w] += 1
		except KeyError:
			result[w] = 1
	return result

def prepare_test_set(test_files):
	test_set = []
	for f in test_files:
		temp = open(f, "r", encoding="utf-8-sig")
		lines = temp.readlines()
		test_set.append([temp.name, utility.extract_annotations(lines[0]), extract_words_with_count(lines[2])])
		temp.close()
	return test_set

def nb_d_class(model, test_files):
	test_set = prepare_test_set(test_files)
	results = nb_d.classify_file_set(model, test_set)
	return results

def knn_class(model, test_files):
	test_set = prepare_test_set(test_files)
	results = knn_f.classify_file_set(model[2], test_set)
	return results
	
def classify(model, test_files, classifier=0):
	c_result = None
	if classifier == 0:
		c_result = nb_d_class(model, test_files)
	elif classifier == 1:
		c_result = knn_class(model, test_files)
	return c_result

	
def print_results(clas_results):
	for res in clas_results:
		print(res)
	
def run_gui(model_name):
	f = open(model_name, "r", encoding="utf-8-sig")
	dump = f.readline()
	extraced = json.loads(dump)
	model = extraced[1]
	classifier = extraced[0]
	
	master = Tk()
	
	enter_t = Label(master, text="Enter text you want to classify:")
	enter_t.pack(fill=X)

	input_t = Text(master)
	input_t.pack(fill=X)
	def callback():
		text = input_t.get(1.0, END)
		text = extract_words_with_count(text)
		res = None
		if classifier == 0:
			res = nb_d.classify_text_only(model, text)
		elif classifier == 1:
			res = knn_f.classify_text_only(model[2],text)
		res_var.set(f"Classified as: {res}")
	
	button = Button(master, text="Ok", command=callback)
	button.pack()

	res_var = StringVar()
	res_var.set("This will change into classification result.")
	result_t = Label(master, textvariable=res_var)
	result_t.pack()

	master.mainloop()


def main():
	parser = argparse.ArgumentParser(description="KIV/UIR semestral work")
	subparsers = parser.add_subparsers(title="we", description="pokus", help="pls work")
	console_parser = subparsers.add_parser("Console", help="blabla")
	gui_parser = subparsers.add_parser("GUI", help="ASDASD")
	console_parser.add_argument("classes_name_file", type=str, help="List of classes lables")
	console_parser.add_argument("train_files", type=str, help="Training files")
	console_parser.add_argument("test_files", type=str, help="Test files")
	console_parser.add_argument("irs", type=str,
		help="Choose information retrieval method: bow = Bag Of Words, bi = Bigram, bowtfidf = Bag Of Words + TF-IDF")
	console_parser.add_argument("classifier", type=str, help="Choose classifier: nb = Naive Bayes, knn = k-Nearest Neighbors")
	console_parser.add_argument("model_name", type=str, help="Model name")

	gui_parser.add_argument("model_name", type=str, help="Model name")

	args = parser.parse_args()

	if len(vars(args)) == 1:
		run_gui(args.model_name)
		return

	if len(vars(args)) <= 0:
		print("No arguments given! Run with -h for help.")
		return

	parametrization = 0
	if args.irs == "bow":
		parametrization = 0
	elif args.irs == "bi":
		parametrization = 1
	elif args.irs == "bowtfidf":
		parametrization = 2
	else:
		print("Urecognized parametrization option. Exiting...")
		return
	
	classificator = 0
	if args.classifier == "nb":
		classificator = 0
	elif args.classifier == "knn":
		classificator = 1
	else:
		print("Unrecognized classificator option. Exiting...")
		return

	model = create_model(glob.glob(args.train_files+"/*.lab"), args.classes_name_file, parametrization, classificator)
	save_model(model, args.model_name, classificator)
	c_results = classify(model, glob.glob(args.test_files+"/*.lab"), classificator)
	print_results(c_results)
	if classificator == 0:
		print(f"Total accuracy: {nb_d.calculate_total_acc(c_results):.2f}%")
	elif classificator == 1:
		print(f"Total accuracy: {knn_f.calculate_total_acc(c_results):.2f}%")

if __name__ == "__main__":
	main()
