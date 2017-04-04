# from __future__ import absolute_import, division, print_function
import codecs
import glob
import string
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from collections import Counter

import numpy as np
from numpy import array

# from nltk.tag.stanford import StanfordNERTagger

from sklearn.feature_extraction.text import CountVectorizer

np.random.seed(69)

# Class for Sentence
class Sentence:

    def setSentenceParams(self, sno, slen):
        self.sno = sno
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

# Make tokens of words in the sentences
def get_tokens(tokens, sent_tokenize_list):
    for sentences in sent_tokenize_list:
        tokens.append(nltk.word_tokenize(sentences))
    return tokens

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
        tempSent = Sentence()
        lDist = 1
        sentNum += 1
        for word in sent:
            tempWord = Word(word)
            tf = freq[tempWord.text]
            # pos_val = pos_array[word]
            # pos = 0
            # if (pos_val == 'NNP'):
            #     pos = 36
            # elif (pos_val == 'NN'):
            #     pos =35
            # elif (pos_val == 'RB'):
            #     pos = 8
            # elif (pos_val == 'JJ'):
            #     pos = 7
            # elif (pos_val == 'IN'):
            #     pos = 6
            # elif (pos_val == 'CC'):
            #     pos = 5
            # elif (pos_val == 'NNS'):
            #     pos = 4
            tempWord.setParams(tf, gDist, lDist, sentNum)
            # tempWord.setPOS(pos)
            wordList.append(tempWord)
            lDist += 1
            gDist += 1
        tempSent.setSentenceParams(sentNum, lDist)
        sentList.append(tempSent)
    return wordList, sentList

# Print word objects
def print_word_objects(wordList):
    for word in wordList:
        print word, word.tf, word.gDist, word.lDist, "\n"

# Print sentence objects
def print_sentence_objects(sentList):
    for sent in sentList:
        parts = raw_data.split('.')
        print parts[sent.sno], sent.sno, sent.slen, "\n"

# For supervised traning set
def output_array_build(wordList):
    sentMat = []

    for word in wordList:
        wordMat = []
        wordMat.append(word.text)
        wordMat.append(word.gDist)
        sentMat.append(wordMat)

    return array(sentMat)

# Make numpy array
def make_numpy_array(wordList):
    sentMat = []

    for word in wordList:
        wordMat = []
        # wordMat.append(word.text)
        wordMat.append(word.tf)
        wordMat.append(word.gDist)
        wordMat.append(word.lDist)
        wordMat.append(word.sentNum)
        # wordMat.append(word.pos)
        # print np.array(wordMat)
        sentMat.append(wordMat)
    # print sentMat
    return sentMat
      
corpus_raw = u""

# Sentence extraction
raw_data = get_input_files(corpus_raw)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# nltk.help.upenn_tagset()

# Tokenize into sentences
raw_sentences = tokenizer.tokenize(raw_data)

# Make a list of sentences
tokens = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        tokens.append(sentence_to_wordlist(raw_sentence))

# Removal of stop words
cleaned_tokens = remove_stopwords(tokens)

# For Counter
cleaned_raw_data = sentence_to_wordlist(raw_data)

# print cleaned_raw_data

# Store term frequency for lookup
freq = Counter(cleaned_raw_data)

# Do POS Tagging
pos_data = nltk.pos_tag(cleaned_raw_data)

# print pos_data[0]

pos_array = {}

for word in pos_data:
    pos_array[word[0]] = word[1]

# print pos_array
# print pos_array['Sharman']

# Create Objects for Word class
wordList, sentList = createWordList(pos_array, freq, cleaned_tokens)

'''# Show word objects
print_word_objects(wordList) 

# Show sentence objects
print_sentence_objects(sentList) 


# For printing the specific sentences which are selected according to the weight
parts = raw_data.split('.')

print parts[10]'''


# Co occurence matrix of words
count_model = CountVectorizer(ngram_range = (1, 1))
X = count_model.fit_transform(cleaned_raw_data)
Xc = (X.T * X)
# print count_model.vocabulary_
# print Xc.shape

# Numpy array
numpy_array = make_numpy_array(wordList)

# print numpy_array

# print numpy_array,  numpy_array.shape

# Output array for supervision
output_array = output_array_build(wordList)

# print type(numpy_array[0][1])

# (p1, p2) = numpy_array.shape
p1 = len(numpy_array)

sent_dict = {}

for i in range(p1):
    if output_array[i][0] in sent_dict:
        pass
    else:
        sent_dict[output_array[i][0]] = output_array[i][1]

# print sent_dict

corpus_summary = u""

summary_data = get_summary_files(corpus_summary)

# print summary_data

# Tokenize into sentences
raw_summaries = tokenizer.tokenize(summary_data)

# Make a list of sentences
tokens_summary = []
for raw_summary in raw_summaries:
    if len(raw_summary) > 0:
        tokens_summary.append(sentence_to_wordlist(raw_summary))

# Removal of stop words
cleaned_tokens_summary = remove_stopwords(tokens_summary)

# set output array for supervised learning
outputMat = []
outputMat[:p1] = [0] * p1
for sent in cleaned_tokens_summary:
    for word in sent:
        # word_mat = []
        if (word in sent_dict):            
            index = sent_dict[word]
            outputMat[int(index)] = 1
            # word_mat.append[outputMat[int(index)]]



# for i in range(len(outputMat)):
#     word_mat = []
#     word_mat.append(outputMat[i])

output_mat = np.array(outputMat)

# print output_mat

op_array = []
for i in output_mat:
    temp = []
    temp.append(i)
    op_array.append(temp)

# print op_array

op_array = np.array(op_array)

# print output_array
# print op_array





###############################################################




'''import tensorflow as tf
import tflearn

# Logical OR operator
X = numpy_array
# print X
Y = op_array

# Graph definition
with tf.Graph().as_default():
    tflearn.init_graph(seed=1)
    g = tflearn.input_data(shape=[None, 4])
    g = tflearn.fully_connected(g, 128, activation='linear')
    # g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 1, activation='sigmoid')
    g = tflearn.regression(g, optimizer='sgd', learning_rate=2.,
                           loss='mean_square')

    # Model training
    m = tflearn.DNN(g)

    train = False

    if(train):
        m.fit(X, Y, n_epoch=100, snapshot_epoch=False)
        m.save('ats.model')
    else:
        m.load('ats.model')

    
    # print numpy_array[8]

    # parts = raw_data.split(' ')

    # print parts[8]

    # print("1 or 1:", m.predict([numpy_array[8]]))
    # print op_array[8]'''



#################################################################

from sklearn.naive_bayes import GaussianNB

X = numpy_array
Y = op_array

print X.shape
print Y.T.shape


Y = Y.T

clf = GaussianNB()
clf.fit(X, Y)

print clf.predict([[1, 46, 46, 2]])