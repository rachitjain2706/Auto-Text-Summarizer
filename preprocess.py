import glob
import re
import os

# Append sentences from all the books
def get_input_files(corpus_raw):
    book_filenames = sorted(glob.glob("/home/rachit/Downloads/Data7/corpus/fulltext/*.xml"))
    for book_filename in book_filenames:
        with open(book_filename, "r") as book_file:
            corpus_raw = book_file.read()
        raw_data = re.sub('<[^>]*>', '', corpus_raw)
        # print os.path.basename(book_filename).split('.')[0]
    	with open("/home/rachit/Downloads/Data7/corpus/training_full_text/" + os.path.basename(book_filename).split('.')[0] + ".txt", "w+") as book_new_file:
    		book_new_file.write(raw_data)
    		book_new_file.close
    return raw_data

# Append sentences from all the books
def get_full_text(corpus_raw):
    book_filenames = sorted(glob.glob("/home/rachit/Downloads/Data7/corpus/citations_class/*.xml"))
    for book_filename in book_filenames:
        with open(book_filename, "r") as book_file:
            corpus_raw = book_file.read()
        raw_data = re.sub('<[^>]*>', '', corpus_raw)
    	with open("/home/rachit/Downloads/Data7/corpus/training_class_2/" + os.path.basename(book_filename).split('.')[0] + ".txt", "w+") as book_new_file:
    		book_new_file.write(raw_data)
    		book_new_file.close
    return raw_data

# Get input data
corpus_raw = u""
raw_data = get_input_files(corpus_raw)
# print raw_data

# Get summary data
corpus_raw = u""
raw_text = get_full_text(corpus_raw)