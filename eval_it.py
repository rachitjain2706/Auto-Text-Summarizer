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

prob = [0.33582089552238803, 0.3180722891566265, 0.24579831932773108, 0.314540059347181, 0.30404463040446306, 0.31962025316455694, 0.17857142857142855, 0.28954937679769893]



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
    book_filenames = sorted(glob.glob("/home/rachit/Music/test_fl/*.txt"))
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
    # if max_nNouns <= 0:
    #     max_nNouns = 1
    for sentence in all_sentences:
        sentence.avg_tfisf /= max_avg_tfsif
        sentence.n_nouns /= max_nNouns
        sentence.slen /= max_sentLen
    return all_sentences

# Get posterior probability
def get_lfeatures(prob_label, prob_flabel, prob_features):
    prob_post = []
    for i in range(len(prob_flabel)):
        temp = prob_flabel[i] / prob_features[i]
        prob_post.append(temp * prob_label)
    return prob_post

# Write summary into a file
def write_summary(summary):
    book_filenames = sorted(glob.glob("/home/rachit/Music/summ/summ.txt"))
    with open(book_filename, "w+") as book_file:
        book_file.write(summary)
    book_file.close
    # return corpus_raw


corpus_raw = u""

# Sentence extraction
raw_data = get_input_files(corpus_raw)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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

# wordList, sentList = createWordList(pos_array, freq, cleaned_tokens)

# Show word objects
# print_word_objects(wordList)

# Show sentence objects
# print_sentence_objects(sentList)

all_sentences, max_avg_tfsif, max_nNouns, max_sentLen = sent_rank(raw_sentences, pos_array, isf_dict)

# print max_avg_tfsif

all_sentences = normalize_shit(all_sentences, max_avg_tfsif, max_nNouns, max_sentLen)
# print len(all_sentences)

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
    avg_tfisf_list.append(sentence.avg_tfisf)
    n_nouns_list.append(sentence.n_nouns)
    slen_list.append(sentence.slen)

med_tfisf = statistics.median(map(float, avg_tfisf_list))
med_n_nouns = statistics.median(map(float, n_nouns_list))
med_slen = statistics.median(map(float, slen_list))

ip_array = []

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
    ip_array.append([i, j, k])

# print len(ip_array)

sent_prob = []

for small_array in ip_array:
    _i = small_array[0]
    _j = small_array[1]
    _k = small_array[2]
    if _i == 0 and _j == 0 and _k == 0:
        temp = prob[0]
    elif _i == 0 and _j == 0 and _k == 1:
        temp = prob[1]
    elif _i == 0 and _j == 1 and _k == 0:
        temp = prob[2]
    elif _i == 0 and _j == 1 and _k == 1:
        temp = prob[3]
    elif _i == 1 and _j == 0 and _k == 0:
        temp = prob[4]
    elif _i == 1 and _j == 0 and _k == 1:
        temp = prob[5]
    elif _i == 1 and _j == 1 and _k == 0:
        temp = prob[6]
    elif _i == 1 and _j == 1 and _k == 1:
        temp = prob[7]

    sent_prob.append(temp)

# print sent_prob

ratio = input("Enter compression ratio: ")

# ratio = 0.24

val = math.ceil(ratio * len(ip_array))
# print val

sent_prob.sort(reverse = True)
sorted_arr = sent_prob

# print sorted_arr

# print val

max_prob = sorted_arr[int(val)]

# print max_prob

summary = u""

print '-------------------------------------------------------------------------------------------------------------'

index = 0
for summ in sent_prob:
    index += 1
    if summ >= max_prob:
        summary += raw_sentences[all_sentences[index].sno]

print '-------------------------------------------------------------------------------------------------------------'



write_summary(summary)