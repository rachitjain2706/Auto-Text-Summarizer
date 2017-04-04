import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import os

import codecs
import glob
import string

import numpy

# text = """To Sherlock Holmes she is always the woman. I have
# seldom heard him mention her under any other name. In his eyes she
# eclipses and predominates the whole of her sex. It was not that he
# felt any emotion akin to love for Irene Adler. All emotions, and that
# one particularly, were abhorrent to his cold, precise but admirably
# balanced mind. He was, I take it, the most perfect reasoning and
# observing machine that the world has seen, but as a lover he would
# have placed himself in a false position. He never spoke of the softer
# passions, save with a gibe and a sneer. They were admirable things for
# the observer-excellent for drawing the veil from mens motives and
# actions. But for the trained reasoner to admit such intrusions into
# his own delicate and finely adjusted temperament was to introduce a
# distracting factor which might throw a doubt upon all his mental
# results. Grit in a sensitive instrument, or a crack in one of his own
# high-power lenses, would not be more disturbing than a strong emotion
# in a nature such as his. And yet there was but one woman to him, and
# that woman was the late Irene Adler, of dubious and questionable
# memory.
# """

text = "Anish is a chutiya. Rachit is Cool. Soham is scared of his mom."


# def get_tokens(text):
#     lowers = text.lower()
#     #remove the punctuation using the character deletion step of translate
#     no_punctuation = lowers.translate(None, string.punctuation)
#     tokens = nltk.word_tokenize(no_punctuation)
#     return tokens

# def remove_stopwords(tokens):
#     filtered = [w for w in tokens if not w in stopwords.words('english')]
#     return filtered

# def stem_tokens(tokens, stemmer):
#     stemmed = []
#     for item in tokens:
#         stemmed.append(stemmer.stem(item))
#     return stemmed


# tokens = get_tokens(text)

# stop_tokens = remove_stopwords(tokens)

# stemmer = PorterStemmer()
# stemmed_tokens = stem_tokens(stop_tokens, stemmer)

# count = Counter(stop_tokens)
# print count

# count = Counter(stemmed_tokens)
# print count

# path = "/home/rachit/Documents/cnn/stories/"
book_filenames = sorted(glob.glob("/home/rachit/Downloads/got/data/*.txt"))


corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()



token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

lowers = corpus_raw.lower()
# no_punctuation = lowers.translate(str.maketrans('', '', string.punctuation))
token_dict[0] = lowers

# print token_dict[0]

tfidf = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english')
tfs = tfidf.fit_transform(token_dict.values())

print tfs

count = Counter(token_dict)
# print count

feature_names = tfidf.get_feature_names()

# print(np.cov(feature_names))

print feature_names

# feature_names = tfidf.get_feature_names()
# for col in r
# print tfs.todense()

# str = "Sherlock Holmes. Observing machine"
# response = tfidf.transform([str])

# print response