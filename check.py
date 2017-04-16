import codecs
import glob
import string
import re

import math # for log operations in tfisf

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from collections import Counter

import numpy as np
from numpy import array
import statistics

import timeit

# Class for Sentence
class Sentence:
    def setSentenceParams(self, n_nouns, avg_tfisf, sno):
        self.n_nouns = n_nouns
        self.avg_tfisf = avg_tfisf
        self.sno = sno

    def setSentLen(self, slen):
        self.slen = slen

# Class for word
class Word:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text

    def setParams(self, tf, gDist, lDist, sentNum):
        self.tf = tf
        self.gDist = gDist
        self.lDist = lDist
        self.sentNum = sentNum

    def setPOS(self, pos):
        self.pos = pos


# Append sentences from all the books
def get_input_files(corpus_raw):
    book_filenames = sorted(glob.glob("/home/rachit/Music/full_text/*.txt"))
    for book_filename in book_filenames:
        # print("Reading '{0}'...".format(book_filename))
        with open(book_filename, "r") as book_file:
            corpus_raw += book_file.read()
        book_file.close
        # print("Corpus is now {0} characters long".format(len(corpus_raw)))
    return corpus_raw

# Append summaries from all the books
def get_summary_files(corpus_raw):
    book_filenames = sorted(glob.glob("/home/rachit/Music/training_class/*.txt"))
    for book_filename in book_filenames:
        # print("Reading '{0}'...".format(book_filename))
        with open(book_filename, "r") as book_file:
            corpus_raw += book_file.read()
        book_file.close
        # print("Corpus is now {0} characters long".format(len(corpus_raw)))
    return corpus_raw

# Make words, only characters kept
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]", " ", raw)
    words = clean.split()
    return words

# Stopword removal
def remove_stopwords(tokens):
    cleaned_tokens = []
    stop_words = stopwords.words('english')
    for token in tokens:
        cleaned_tokens_sentence = []
        for word in token:
            if word not in stop_words:
                cleaned_tokens_sentence.append(word)
        cleaned_tokens.append(cleaned_tokens_sentence)
    return cleaned_tokens

# Making word list
def createWordList(pos_array, freq, cleaned_tokens):
    sentNum = 1
    gDist = 1
    wordList = []
    sentList = []
    for sent in cleaned_tokens:
        # print sent
        tempSent = Sentence()
        lDist = 1
        sentNum += 1
        for word in sent:
            tempWord = Word(word)
            tf = freq[tempWord.text]
            pos_val = pos_array[word]
            pos = 0
            if (pos_val == 'NNP' or pos_val == 'NN' or pos_val == 'NNS' or pos_val == 'NNPS'):
                pos = 1
            tempWord.setParams(tf, gDist, lDist, sentNum)
            tempWord.setPOS(pos)
            wordList.append(tempWord)
            lDist += 1
            gDist += 1
        # tempSent.setSentenceParams(sentNum, lDist)
        sentList.append(tempSent)
    return wordList, sentList

# Print word objects
def print_word_objects(wordList):
    for word in wordList:
        print word, word.tf, word.gDist, word.lDist, word.pos, "\n"

# Print sentence objects
def print_sentence_objects(sentList):
    for sent in sentList:
        parts = raw_data.split('.')
        # print sent
        print sent.sno, sent.slen, "\n"
        # for word in sent:
            # print word, 'lowl'
        # print sent
        
        # print 'lowl'

# Stopword removal
'''def remove_stopwords_array(tokens):
    stop_words = stopwords.words('english')
    cleaned_tokens_sentence = []
    for word in tokens:
        # print word
        if word not in stop_words:
            cleaned_tokens_sentence += word
        # print cleaned_tokens_sentence
    return cleaned_tokens_sentence'''

# def tfisf_calculation(tokens, freq):
#     for sent in tokens:
#         tfisf = 0
#         n_words = 1
#         for word in sent:
#             n_words += 1
#             for tempSent in tokens:
#                 counter = 1
#                 for tempWord in tempSent:
#                     if (word.lower() == tempWord.lower()):
#                         counter += 1
#                         break
#             isf = 1 / counter
#             tfisf += freq[word] * isf
#         av_tfisf = tfisf / n_words
        # print sent, av_tfisf

def sent_rank(raw_sentences, pos_array, isf_dict):
    sentNum = 0
    all_sentences = []
    max_avg_tfsif = -1
    max_nNouns = -1
    max_sentLen = -1
    for sent in raw_sentences:
        tempSent = Sentence()
        sentNum += 1
        tfisf = 0
        pos = 0
        for word in sent:
            if (word in pos_array):
                tempWord = Word(word)
                pos_val = pos_array[word]                
                if (pos_val == 'NNP' or pos_val == 'NNPS'):
                    pos += 1
                if (word in isf_dict):
                    tfisf += isf_dict[word]
        avg_tfisf = float(tfisf) / len(sent)

        if(avg_tfisf > max_avg_tfsif): # For normalizing
            max_avg_tfsif = avg_tfisf

        if(pos > max_nNouns):   #For normlizing
            max_nNouns = pos

        if(len(sent) > max_sentLen):
            max_sentLen = len(sent)

        tempSent.setSentenceParams(float(pos), avg_tfisf, sentNum) 
        tempSent.setSentLen(float(len(sent)))       
        all_sentences.append(tempSent)
    # print max_avg_tfsif

    return all_sentences, max_avg_tfsif, max_nNouns, max_sentLen

def ISF(N, n):
    '''N : total number of senteces in corpus
       n : number of sentences with our word in it'''
    return float(math.log(float(N)/n) + 1)

def make_tfisf_dict(raw_sentences, raw_data, freq):
    n_sents = len(raw_sentences)  # This is our N

    unique_words = set(raw_data.split())

    # print unique_words

    final_list = []
    sent_occurence_counter = 0
    # calculating number of sentences with our word in it
    for unq_word in unique_words:
        for sent in raw_sentences:
            for word in sent.split():
                if unq_word == word:
                    sent_occurence_counter += 1
                    break
        final_list.append([unq_word, freq[unq_word]*ISF(n_sents, sent_occurence_counter)])
        sent_occurence_counter = 0
    # print final_list

    isf_dict={}

    for word in final_list:
        isf_dict[word[0]] = word[1]

    return isf_dict

def normalize_shit(all_sentences, max_avg_tfsif, max_nNouns, max_sentLen):
    for sentence in all_sentences:
        sentence.avg_tfisf /= max_avg_tfsif
        sentence.n_nouns /= max_nNouns
        sentence.slen /= max_sentLen
    return all_sentences

# Get posterior probability
def get_lfeatures(prob_label, prob_flabel, prob_features):
    # return prob_label * prob_flabel / prob_features
    prob_post = []
    for i in range(len(prob_flabel)):
        temp = prob_flabel[i] / prob_features[i]
        prob_post.append(temp * prob_label)
    return prob_post

start = timeit.default_timer()

corpus_raw = u""

# Sentence extraction
raw_data = get_input_files(corpus_raw)

# Tokenize into sentences
# raw_sentences = tokenizer.tokenize(raw_data)
raw_sentences = nltk.sent_tokenize(raw_data)

# print raw_sentences

# Make a list of sentences
tokens = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        tokens.append(sentence_to_wordlist(raw_sentence))

# print tokens

# Removal of stop words
cleaned_tokens = remove_stopwords(tokens)

# print cleaned_tokens

# For Counter
# Make an array of sentences
tokens_array = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        tokens_array += sentence_to_wordlist(raw_sentence)

# print tokens_array

# Removal of stop words, For Counter
cleaned_raw_data = sentence_to_wordlist(raw_data)

# print cleaned_raw_data

# Store term frequency for lookup
freq = Counter(cleaned_raw_data)

# Make tf-isf dict
isf_dict = make_tfisf_dict(raw_sentences, raw_data, freq)

# print isf_dict

# Do POS Tagging
pos_data = nltk.pos_tag(cleaned_raw_data)

# print cleaned_raw_data[25]

# print pos_data[0]

pos_array = {}

for word in pos_data:
    pos_array[word[0]] = word[1]

wordList, sentList = createWordList(pos_array, freq, cleaned_tokens)

# Show word objects
# print_word_objects(wordList)

# Show sentence objects
# print_sentence_objects(sentList)

all_sentences, max_avg_tfsif, max_nNouns, max_sentLen = sent_rank(raw_sentences, pos_array, isf_dict)

# print max_avg_tfsif

all_sentences = normalize_shit(all_sentences, max_avg_tfsif, max_nNouns, max_sentLen)

# print all_sentences

# print len(all_sentences)

'''ip_array = []
for sentence in all_sentences:
    ip_array.append([sentence.avg_tfisf, sentence.n_nouns, sentence.slen])'''

# print all_sentences.avg_tfisf
# print median(all_sentences.avg_tfisf)
avg_tfisf_list = []
n_nouns_list = []
slen_list = []
for sentence in all_sentences:
    # print statistics.median(map(float, sentence.avg_tfisf))
    avg_tfisf_list.append(sentence.avg_tfisf)
    n_nouns_list.append(sentence.n_nouns)
    slen_list.append(sentence.slen)

med_tfisf = statistics.median(map(float, avg_tfisf_list))
med_n_nouns = statistics.median(map(float, n_nouns_list))
med_slen = statistics.median(map(float, slen_list))

ip_array = []

_zero = 0
_one = 0
_two = 0
_three = 0
_four = 0
_five = 0
_six = 0
_seven = 0

for sentence in all_sentences:
    i = 0
    j = 0
    k = 0
    if sentence.avg_tfisf > med_tfisf:
        i = 1
    if sentence.n_nouns > med_n_nouns:
        j = 1
    if sentence.slen > med_slen:
        k = 1
    if i == 0 and j == 0 and k == 0:
        _zero += 1
    if i == 0 and j == 0 and k == 1:
        _one += 1
    if i == 0 and j == 1 and k == 0:
        _two += 1
    if i == 0 and j == 1 and k == 1:
        _three += 1
    if i == 1 and j == 0 and k == 0:
        _four += 1
    if i == 1 and j == 0 and k == 1:
        _five += 1
    if i == 1 and j == 1 and k == 0:
        _six += 1
    if i == 1 and j == 1 and k == 1:
        _seven += 1
    ip_array.append([i, j, k])

features_list = [_zero, _one, _two, _three, _four, _five, _six, _seven]

# print features_list

# print ip_array

# print all_sentences

corpus_summary = u""

summary_data = get_summary_files(corpus_summary)

# print summary_data

# Tokenize into sentences
# raw_summaries = tokenizer.tokenize(summary_data)
raw_summaries = nltk.sent_tokenize(summary_data)

# print raw_sentences

# Check if sentence in summary
'''p1 = len(raw_sentences)
outputMat = []
outputMat[:p1] = [0] * p1
for summary_sentence in raw_summaries:
    index = 0    
    for raw_sentence in raw_sentences:        
        index += 1
        if (summary_sentence == raw_sentence):
            # print index, '->', summary_sentence, '->', raw_sentence, '\n\n\n'
            outputMat[index] = 1
            # print index'''

ones = 0
index = -1
p1 = len(raw_sentences)
# print p1
outputMat = []
outputMat[:p1] = [0] * p1

for raw_sentence in raw_sentences:
    index += 1
    # for summary_sentence in raw_summaries:
    if (raw_sentence in raw_summaries):
        # print index, '->', summary_sentence, '->', raw_sentence, '\n\n\n'
        ones += 1
        outputMat[index] = 1

# print ones
prob_label = float(ones) / p1

# print prob_label

prob_features = []

for i in features_list:
    prob_features.append(float(i) / p1)

# print prob_features

__zero = 0
__one = 0
__two = 0
__three = 0
__four = 0
__five = 0
__six = 0
__seven = 0

for i, sentence in enumerate(all_sentences):
    if outputMat[i] == 1:
        _i = ip_array[i][0]        
        _j = ip_array[i][1]
        _k = ip_array[i][2]
        if _i == 0 and _j == 0 and _k == 0:
            __zero += 1
        if _i == 0 and _j == 0 and _k == 1:
            __one += 1
        if _i == 0 and _j == 1 and _k == 0:
            __two += 1
        if _i == 0 and _j == 1 and _k == 1:
            __three += 1
        if _i == 1 and _j == 0 and _k == 0:
           __four += 1
        if _i == 1 and _j == 0 and _k == 1:
            __five += 1
        if _i == 1 and _j == 1 and _k == 0:
            __six += 1
        if _i == 1 and _j == 1 and _k == 1:
            __seven += 1

flabel_list = [__zero, __one, __two, __three, __four, __five, __six, __seven]

# print flabel_list

prob_flabel = []

for i in flabel_list:
    prob_flabel.append(float(i) / sum(flabel_list))

# print sum(prob_flabel)

prob_post = get_lfeatures(prob_label, prob_flabel, prob_features)

print prob_post

stop = timeit.default_timer()


print 'Time  = ', stop - start