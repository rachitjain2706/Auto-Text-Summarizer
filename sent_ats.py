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
    book_filenames = sorted(glob.glob("/home/rachit/Music/traning_class/*.txt"))
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


corpus_raw = u""

# Sentence extraction
raw_data = get_input_files(corpus_raw)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Tokenize into sentences
raw_sentences = tokenizer.tokenize(raw_data)

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

ip_array = []
for sentence in all_sentences:
    ip_array.append([sentence.avg_tfisf, sentence.n_nouns, sentence.slen])

# print ip_array

# print all_sentences

corpus_summary = u""

summary_data = get_summary_files(corpus_summary)

# print summary_data

# Tokenize into sentences
raw_summaries = tokenizer.tokenize(summary_data)

# print raw_sentences

# Check if sentence in summary
p1 = len(raw_sentences)
outputMat = []
outputMat[:p1] = [0] * p1
for summary_sentence in raw_summaries:
    index = 0    
    for raw_sentence in raw_sentences:        
        index += 1
        if (summary_sentence == raw_sentence):
            # print index, '->', summary_sentence, '->', raw_sentence, '\n\n\n'
            outputMat[index] = 1
            # print index

# print len(outputMat)

op_array = []
for val in outputMat:
    op_array.append([val])

j = []

# print outputMat[883]

for i in range(0, p1-1):
    y = all_sentences[i].sno
    # print y, '\n'
    # if (outputMat[y] == 1):
        # print raw_sentences[y], all_sentences[i].avg_tfisf, all_sentences[i].n_nouns
    j.append([all_sentences[i].avg_tfisf, all_sentences[i].n_nouns, all_sentences[i].slen])


# print j
# print op_array

# print outputMat[0:50]

'''for sent in all_sentences:
    if (sent.sno < 3):
        print raw_sentences[sent.sno]'''

# ------------------------------------------------------------------------------------ #

import tensorflow as tf
import tflearn

# Logical OR operator
X = ip_array
Y = op_array

# print Y

# Graph definition
with tf.Graph().as_default():
    tflearn.init_graph(seed=1)
    g = tflearn.input_data(shape=[None, 3])
    g = tflearn.fully_connected(g, 20, activation='linear')
    g = tflearn.fully_connected(g, 20, activation='linear')
    g = tflearn.fully_connected(g, 1, activation='sigmoid')
    g = tflearn.regression(g, optimizer='sgd', learning_rate=0.3,
                           loss='mean_square')

    # Model training
    m = tflearn.DNN(g)

    train = True

    if(train):
        m.fit(X, Y, n_epoch=900, snapshot_epoch=False)
        m.save('ats.model')
    else:
        m.load('ats.model')

    # print("To be or not to be:", m.predict([[115.679916309, 17]]))

    for small_array in j:
        # print small_array[0], small_array[1]
        print("To be or not to be:", m.predict([[small_array[0], small_array[1], small_array[2]]]))

    
    # print numpy_array[8]

    # parts = raw_data.split(' ')

    # print parts[8]

    # print("1 or 1:", m.predict([[229, 1]]))
    # for i in xrange(1000):
    #     print("i : ", i , m.predict([[i, 1]]))
    # print op_array[8]'''