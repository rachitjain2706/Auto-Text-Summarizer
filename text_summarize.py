import codecs
import glob
import string
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from collections import Counter

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

# Append sentences from all the books
def get_files(corpus_raw):
    book_filenames = sorted(glob.glob("/home/rachit/Downloads/got/data/*.txt"))
    for book_filename in book_filenames:
        print("Reading '{0}'...".format(book_filename))
        with codecs.open(book_filename, "r", "utf-8") as book_file:
            corpus_raw += book_file.read()
        print("Corpus is now {0} characters long".format(len(corpus_raw)))
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
def createWordList(freq, cleaned_tokens):
    sentNum = 1
    gDist = 1
    wordList = []
    for sent in cleaned_tokens:
        lDist = 1
        sentNum += 1
        for word in sent:
            tempWord = Word(word)
            tf = freq[tempWord.text]
            tempWord.setParams(tf, gDist, lDist, sentNum)
            wordList.append(tempWord)
            lDist += 1
            gDist += 1
    return wordList
        
corpus_raw = u""

# Sentence extraction
raw_data = get_files(corpus_raw)

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

# Store term frequency for lookup
freq = Counter(cleaned_raw_data)

# Create Objects for Word class
wordList = createWordList(freq, cleaned_tokens)

# for word in wordList:
#     print word, word.tf, word.gDist, "\n"

# print nltk.pos_tag(nltk.word_tokenize('BBC and CNN are reported pussy. I have a big,black,hairy pussy cat and a Dog called brandy who is dead.'))