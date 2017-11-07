# TL;DR Auto-Text-Summarizer

The project uses a Naive Bayes approach to classify sentence as summary sentence or not.

**For a detailed report read: `/TLDR.pdf`**

**Data Set:** Australian legal cases taken from the UCI repository(https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports)

We chose to extract certain features from a sentence in order to represent it in a vector/representable space.

We suggest a Naïve Bayes approach to Automatic Text Summarization(ATS) using 
1. term frequency-inverse sentence frequency(tf-isf)
2. sentence length (and)
3. number of nouns in a sentence.

We developed some non Naive Bayes codes as well: NeuralNet.py and TextRank.py (now in /Experimental Code)

If you need help, feel free to ask/make an issue.
