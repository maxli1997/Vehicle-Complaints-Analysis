#Install and load packages 
import nltk
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import re
import numpy as np
import pandas as pd
from pprint import pprint

np.random.seed = 0
import random
random.seed(0) 

# spacy for lemmatization
import spacy

df = pd.read_csv("articlesf.csv", encoding='UTF-8')

# preporcessing for topic modeling
#textdata = df.text
textdata = df.Abstract
data = textdata 
# Remove urls
data = [re.sub(r'https\S+', '', sent) for sent in data]
data = [re.sub(r'http\S+', '', sent) for sent in data]
# Remove new line characters
data = [sent.replace("\n", "") for sent in data]

#Tokenize words and Clean-up text
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence).encode('utf-8'), deacc=True, min_len = 1, max_len = 30))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    stop_words.append('vehicle')
    stop_words.append('car')
    stop_words.append('brake')
    stop_words.append('TL* THE CONTACT')
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
#data_lemmatized = lemmatization(data_words_bigrams)
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

from gsdmm import MovieGroupProcess # move the folder gsdmm from gsdmm-master to be under content directly
mgp = MovieGroupProcess(K= 7, alpha=0.1, beta=0.1, n_iters=30)

docs = texts # the texts should be preprocessed, including tokenization, lemmitization, removing stop words etc.
vocab = set(x for doc in docs for x in doc)
n_terms = len(vocab)
n_docs = len(docs)
y = mgp.fit(docs, n_terms)

threshold = 5 # each topic needs at least five issues
cluster_doc_count = np.array(mgp.cluster_doc_count)
doc_count = np.argsort(cluster_doc_count)
doc_count_des = doc_count[::-1]

index = np.where(cluster_doc_count > threshold)
num_of_topic = np.shape(index)
num_of_topic = num_of_topic[1]
doc_count_des = doc_count_des[0:num_of_topic]

# get the probability of each sentence associated with the most likely topics
prob = []
for i in range(len(docs)):
  prob.append(mgp.choose_best_label(docs[i]))
#prob print(prob) to check what it is

# Get the top 10 sentences in each topic
f = open("GSDMM_results.txt", "w")
prob = np.array(prob)
count = 0
for i in range(num_of_topic):
  index1 = np.where(prob==doc_count_des[i])
  temp = prob[index1[0]]
  index2 = np.argsort(temp[:,1])
  index3 = index2[::-1]
  index4 = []
  index1 = np.array(index1)

  if len(index3)<2415:
    for i in range(len(index3)):
      index4.append(index1[0,index3[i]])
  else:
    for i in range(2415):
      index4.append(index1[0,index3[i]])
  count = count + 1
  f.write(f"Topic {count}:\n The number of posts = {len(index3):.0f} and percentage = {len(index3)/len(docs)*100:.1f}\n")
  #print('The number of posts = %.0f and percentage = %.1f' % (len(index3), len(index3)/len(docs)*100))
  print('the top sentences in this topic')
  for i in index4:
    f.write(textdata[i]+'\n')
  f.write('\n')

f.close()
  # check if you can get the key words in each topic, but read the sentences in each topic 
  # to interpret the topic yourself