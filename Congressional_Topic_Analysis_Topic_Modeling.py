# -*- coding: utf-8 -*-
"""
Toni Hanrahan
Congressional Topic Analysis – Topic Modeling

"""


# -*- coding: utf-8 -*-

###################################################
##
## LDA for Topic Modeling
##
###################################################


#%%



#combine text files into one

import fileinput
import glob

file_list = glob.glob("*.txt")

with open('result.txt', 'w') as file:
    input_lines = fileinput.input(file_list)
    file.writelines(input_lines)


#read in the single combined text file

import pandas as pd

data_combo=pd.read_csv('result_cln.txt', sep=None, header=None, encoding = "ISO-8859-1", error_bad_lines=False);
print(data_combo.head())


#create an index column

data_combo['text'] = data_combo.index

data_combo.index = range(len(data_combo))

print(data_combo.columns)





#documents = data_text
documents = data_combo
print(documents)

print("The length of the file - or number of docs is", len(documents))
print(documents[:5])


#%%
###################################################
###
### Data Prep and Pre-processing
###
###################################################
#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python

import gensim
## IMPORTANT - you must install gensim first ##
## conda install -c anaconda gensim
# conda install -c conda-forge gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer

import numpy as np
np.random.seed(2018)

import nltk
nltk.download('wordnet')
from nltk import PorterStemmer
from nltk.stem import PorterStemmer 
stemmer = PorterStemmer()

from nltk.tokenize import word_tokenize 
from nltk.stem.porter import *

#NOTES
##### Installing gensim caused my Spyder IDE no fail and no re-open
## I used two things and did a restart
## 1) in cmd (if PC)  psyder --reset
## 2) in cmd (if PC) conda upgrade qt

######################################
## function to perform lemmatize and stem preprocessing
############################################################
## Function 1
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

## Function 2
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

##############  try to get this to work ########
#Select a document to preview after preprocessing
doc_sample = documents[documents.text == 50].values[0][0]
print(doc_sample)
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

#%%

## Preprocess the headline text, saving the results as ‘processed_docs’
processed_docs = documents['text'].map(preprocess)
print(processed_docs[:10])

#%%

## Create a dictionary from ‘processed_docs’ containing the 
## number of times a word appears in the training set.

dictionary = gensim.corpora.Dictionary(processed_docs)

## Take a look ...you can set count to any number of items to see
## break will stop the loop when count gets to your determined value
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 5:
        break
    
    
#%%    
#print(processed_docs)   
## Filter out tokens that appear in
## - - less than 15 documents (absolute number) or
## - - more than 0.5 documents (fraction of total corpus size, not absolute number).
## - - after the above two steps, keep only the first 100000 most frequent tokens
 ############## NOTE - this line of code did not work with my small sample
## as it created blank lists.....       
#dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

for doc in processed_docs:
    print(doc)

print(dictionary)

#%%
#######################
## For each document we create a dictionary reporting how many
##words and how many times those words appear. Save this to ‘bow_corpus’
##############################################################################
#### bow: Bag Of Words
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print(bow_corpus[3:5])


#%%
#################################################################
### TF-IDF
#################################################################
##Create tf-idf model object using models.TfidfModel on ‘bow_corpus’ 
## and save it to ‘tfidf’, then apply transformation to the entire 
## corpus and call it ‘corpus_tfidf’. Finally we preview TF-IDF 
## scores for our first document.

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
## pprint is pretty print
from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    ## the break will stop it after the first doc
    break

#%%

#############################################################
### Running LDA using Bag of Words
#################################################################
#According to political scientists, there are usually 40-50 common topics going on in each Congress
    #run once with 40 topics, then again wiht 50
# ~ 12 minutes
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=50, id2word=dictionary, passes=10)
    
# Print the Keyword in the 40 or 50 topics
pprint(lda_model.print_topics())

#doc_lda = lda_model[bow_corpus]


# Compute Perplexity
perplx = lda_model.log_perplexity(bow_corpus)
print('\nPerplexity: ', perplx )  # a measure of how good the model is. lower the better.

#%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Compute Coherence Score
from gensim.models import CoherenceModel
coherence_model_lda = CoherenceModel(model=lda_model, texts=bow_corpus, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

#%%

import pyLDAvis.sklearn as LDAvis
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt
## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.show(vis)

#%%
