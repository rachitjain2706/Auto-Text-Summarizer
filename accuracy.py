import os
import codecs
import glob
import string
import math # for log operations in tfisf
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from collections import Counter

import numpy as np
from numpy import array
import statistics

# Append sentences from all the books
def get_input_files(corpus_raw):
    book_filenames = glob.glob("/home/rachit/Music/test_cc/*.txt")
    list_raw = []
    for book_filename in book_filenames:
        list_raw.append(book_filename)
    return list_raw

# Append sentences from all the books
def get_summ(corpus_raw):
    book_filenames = glob.glob("/home/rachit/Music/summ/*.txt")
    list_raw_summ = []
    for book_filename in book_filenames:
        list_raw_summ.append(book_filename)
    return list_raw_summ

def open_file(list_raw, list_raw_summ):
    scores = []
    for i in xrange(0, len(list_raw)):
        with open(list_raw[i], "r") as book_file:
            corpus_raw = book_file.read()
        # print book_file
        book_file.close        
        with open(list_raw_summ[i], "r") as book_file:
            corpus_summ = book_file.read()
        book_file.close

        scores.append(find_accuracy(corpus_raw, corpus_summ))
    return scores

def find_accuracy(raw_data, raw_summ):

    # Tokenize into sentences
    raw_sentences = nltk.sent_tokenize(raw_data)

    # Tokenize into sentences
    raw_summaries = nltk.sent_tokenize(raw_summ)

    tp = 1
    fp = 1
    fn = 1

    # print raw_sentences
    # print raw_summaries

    for raw_sentence in raw_sentences:
        if raw_sentence in raw_summaries:
            tp += 1
        if raw_sentence not in raw_summaries:
            fn += 1

    for raw_summary in raw_summaries:
        if raw_summary not in raw_sentences:
            fp += 1

    prob_prec = float(tp) / (tp + fp)

    prob_recall = float(tp) / (tp + fn)

    f1score = 2 * (prob_prec * prob_recall) / (prob_prec + prob_recall)

    # s = 0
    # counter = 0
    if f1score > 0.2:
        # counter += 1
        # print tp, fp, fn
        print '-----------------------------------------------------------------------------------------'
        print 'F1 Score = ', f1score
        print 'Precision = ', prob_prec
        print 'Recall = ', prob_recall
        print '-----------------------------------------------------------------------------------------'
        return (f1score)
    else:
        return 0



corpus_raw = u""
corpus_summary = u""

# Sentence extraction
list_raw = get_input_files(corpus_raw)
list_raw_summ = get_summ(corpus_summary)

scores = open_file(list_raw, list_raw_summ)

# print scores

s = 0
index = 0
for i in scores:
    s += i
    if i != 0:
        index += 1

avg_accuracy = float(s) / index


print 'Average accuracy = ', avg_accuracy
