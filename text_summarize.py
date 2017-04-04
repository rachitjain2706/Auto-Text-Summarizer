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
    book_filenames = sorted(glob.glob("/home/rachit/Downloads/Data7/corpus/training_full_text/*.txt"))
    for book_filename in book_filenames:
        # print("Reading '{0}'...".format(book_filename))
        with codecs.open(book_filename, "r", "utf-8") as book_file:
            corpus_raw += book_file.read()
        # print("Corpus is now {0} characters long".format(len(corpus_raw)))
    return corpus_raw

# Append summaries from all the books
def get_summary_files(corpus_raw):
    book_filenames = sorted(glob.glob("/home/rachit/Downloads/Data7/corpus/training_class_2/*.txt"))
    for book_filename in book_filenames:
        # print("Reading '{0}'...".format(book_filename))
        with codecs.open(book_filename, "r", "utf-8") as book_file:
            corpus_raw += book_file.read()
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
def createWordList(pos_data, freq, cleaned_tokens):
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
            # print tempWord.text
            # a = "u\'"
            # index = a + tempWord.text
            # pos = pos_data[a]
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
def output_array(wordList):
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
        sentMat.append(wordMat)

    return np.array(sentMat)
      
corpus_raw = u""

# Sentence extraction
raw_data = get_input_files(corpus_raw)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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

# print pos_data[0][1]

'''pos_array = []

for word in pos_data:
    pos_array[word[0]] = word[1]

print pos_array'''

# Create Objects for Word class
wordList, sentList = createWordList(pos_data, freq, cleaned_tokens)

'''# Show word objects
print_word_objects(wordList) '''

'''# Show sentence objects
print_sentence_objects(sentList) '''

'''
# For printing the specific sentences which are selected according to the weight
parts = raw_data.split('.')

print parts[10] '''


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
output_array = output_array(wordList)

# print type(numpy_array[0][1])

(p1, p2) = numpy_array.shape

sent_dict = {}

for i in range(p1):
    if output_array[i][0] in sent_dict:
        pass
    else:
        sent_dict[output_array[i][0]] = output_array[i][1]

# print sent_dict

corpus_summary = u""

summary_data = get_summary_files(corpus_summary)

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

op_array = []
for i in output_mat:
    temp = []
    temp.append(i)
    op_array.append(temp)

op_array = np.array(op_array)

# print op_array

# print op_array
# print array(outputMat).shape


##################################



class BackPropagationNetwork:
    """A back-propagation network"""

    def __init__(self, layerSize, layerFunctions=None):# layerSize is a tuple of layers
        """Initialize the network"""
        
        self.layerCount = 0
        self.shape = None
        self.weights = []   # weights assigned to a layer are the weights that precede it
                            # so the weights that feed into the layer are the ones which are assigned to it
        
        # Layer info
        self.layerCount = len(layerSize) - 1 # input layer is only a placeholder kinda thing for inputs. So numLayer is 1 less than that
        self.shape = layerSize
        
        # Input Outpur data from last Run
        self._layerInput = []
        self._layerOutput = []

        # layerSize[:-1] All but the last one
        # layerSize[1:] from the first one

        # Create the weight arrays
        for (l1,l2) in zip(layerSize[:-1], layerSize[1:]):#(l1,l2) become for say layersize = (2,3,4) : [(2,3),(3,4)]
            self.weights.append(np.random.normal(scale=0.01, size = (l2, l1+1)))# +1 for the bias node
                                                                                # Inputs are 3x4 so weights will be 4xp

def nonlin(x, deriv=False):
    if(deriv==True):
        return 0.03*(x*(1-x))
    return 1/(1+np.exp(-x))

if __name__ == "__main__":
    n_hiddenlayers = 1
    n_hiddenlayer_neurons = 3
    n_datapoints  = p1
    n_features = p2

    # print p1, p2, "\n"


    # InputData = makeRandomArray(n_datapoints, n_features) # Get Data here!
    # OutputData = makeRandomArray(n_datapoints,1) # Get Data here!

    InputData = numpy_array
    # print InputData.shape
    OutputData = op_array
    # print OutputData.shape[0]


    bpn = BackPropagationNetwork((n_features,n_hiddenlayer_neurons,1))
    weights = bpn.weights
    syn0,syn1 = weights[0],weights[1]

    # print "INPUT DATA:    \n",InputData.shape,"\n",InputData
    
    for j in xrange(60000):

        # print "Syn 0 :    \n",syn0.shape,"\n",syn0

        # print "Syn 1 :    \n",syn1.shape,"\n",syn1 

        l0 = InputData
        l0 = np.vstack([l0.T, np.ones(n_datapoints)]).T

        l1 = nonlin(np.dot(l0, syn0.T))
        l1 = np.vstack((l1.T, np.ones(n_datapoints))).T
        l2 = nonlin(np.dot(l1, syn1.T))

        l2_error = OutputData - l2

        if(j % 10000) == 0:   # Only print the error every 10 steps, to save time and limit the amount of output. 
            print("Error: " + str(np.mean(np.abs(l2_error))))

        l2_delta = l2_error*nonlin(l2, deriv=True)
        # print l2_delta, "\n\n\n"
        l1_error = l2_delta.dot(syn1)
        # print l1_error, "\n\n\n"
        l1_delta = l1_error * nonlin(l1,deriv=True)
        syn1_delta = l1.T.dot(l2_delta).T
        syn1 += syn1_delta

        l1_delta = l1_delta.T
        l1_delta = np.delete(l1_delta, -1, 0).T
        syn0_delta = l0.T.dot(l1_delta).T
        syn0 += syn0_delta

        # print syn0_delta,syn0_delta.shape


        # syn1 = syn1 + 
        # syn1 = syn1.T +  l1.T.dot(l2_delta)
        # syn0 = l0.T.dot(l1_delta)


'''class BackPropagationNetwork:
    """A back-propagation network"""

    def __init__(self, layerSize, layerFunctions=None):# layerSize is a tuple of layers
        """Initialize the network"""
        
        self.layerCount = 0
        self.shape = None
        self.weights = []   # weights assigned to a layer are the weights that precede it
                            # so the weights that feed into the layer are the ones which are assigned to it
        
        # Layer info
        self.layerCount = len(layerSize) - 1 # input layer is only a placeholder kinda thing for inputs. So numLayer is 1 less than that
        self.shape = layerSize
        
        # Input Outpur data from last Run
        self._layerInput = []
        self._layerOutput = []

        # layerSize[:-1] All but the last one
        # layerSize[1:] from the first one

        # Create the weight arrays
        for (l1,l2) in zip(layerSize[:-1], layerSize[1:]):#(l1,l2) become for say layersize = (2,3,4) : [(2,3),(3,4)]
            self.weights.append(np.random.normal(scale=0.1, size = (l2, l1)))# +1 for the bias node
                                                                                # Inputs are 3x4 so weights will be 4xp

def nonlin(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

def makeRandomArray(n, m):
    arr = 2*np.random.random((n,m)) - 1
    return arr

n_hiddenlayers = 1
n_hiddenlayer_neurons = 3
n_datapoints  = p1
n_features = p2

InputData = numpy_array
OutputData = op_array

bpn = BackPropagationNetwork((n_features,n_hiddenlayer_neurons,1))

weights = bpn.weights

syn0,syn1 = weights[0].T,weights[1].T

print "INPUT DATA:  \n",InputData.shape,"\n",InputData

print "Syn 0 :  \n",syn0.shape,"\n",syn0

print "Syn 1 :  \n",syn1.shape,"\n",syn1


for j in xrange(600):  
    
    # Calculate forward through the network

    l0 = InputData
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    
    # Back propagation of errors using the chain rule. 
    l2_error = OutputData - l2
    if(j % 10) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output. 
        print("Error: " + str(np.mean(np.abs(l2_error))))
        
    l2_delta = l2_error*nonlin(l2, deriv=True)
    
    l1_error = l2_delta.dot(syn1.T)
    
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
    #update weights (no learning rate term)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

# print syn0'''