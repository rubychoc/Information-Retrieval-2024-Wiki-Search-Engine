import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import threading
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from pathlib import Path
import pickle
import pandas as pd
from functools import partial
from google.cloud import storage
import concurrent.futures
import math
import gzip
import csv
import io
from gensim.models import KeyedVectors
import gensim.downloader as api
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

nltk.download('stopwords')


bucket_name = 'ruby_bucket_327064358'
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

# load body index
file_path = 'body_index_with_stem/body_index_with_stem.pkl'
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
body_index = pickle.loads(contents)


file_path = 'global_dics/docid_title_pairs.pkl'
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
doc_title_pairs = pickle.loads(contents)


file_path = 'global_dics/doc_lengths_body.pkl'
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
doc_lengths = pickle.loads(contents)


file_path = 'global_dics/idf_scores_body_with_stem.pkl'
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
idf_scores_body = pickle.loads(contents)

# file_path = 'global_dics/normalization_factor_per_doc_body.pkl'
# blob = bucket.blob(file_path)
# contents = blob.download_as_bytes()
# normalization_factor_per_doc_body = pickle.loads(contents)

# load title index
file_path = 'title_index/title_index.pkl'
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
title_index = pickle.loads(contents)

# load title index
file_path = 'global_dics/title_lengths.pkl'
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
title_lengths = pickle.loads(contents)

file_path = 'global_dics/idf_scores_title.pkl'
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
idf_scores_title = pickle.loads(contents)

# file_path = 'global_dics/normalization_factor_per_title.pkl'
# blob = bucket.blob(file_path)
# contents = blob.download_as_bytes()
# normalization_factor_per_title = pickle.loads(contents)

#load page rank
file_path = 'global_dics/page_rank.pkl'
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
page_rank = pickle.loads(contents)

# load page views
file_path = 'global_dics/page_views.pkl'
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
page_views = pickle.loads(contents)


# load word2vec
file_name = "glove.6B.50d.txt"

# Define the local file path where you want to download the file
local_glove_file_path = "glove.6B.50d.txt"
local_word2vec_file_path = "glove.6B.50d.txt"

# Download the GloVe model file from GCS bucket
blob = bucket.blob(file_name)
blob.download_to_filename(local_glove_file_path)

# Convert the GloVe model file to Word2Vec format
glove_file = local_glove_file_path
tmp_file = local_word2vec_file_path
_ = glove2word2vec(glove_file, tmp_file)

# Load the model using the Gensim API's load function
model = KeyedVectors.load_word2vec_format(tmp_file)

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)

filter_func = lambda tok: tok not in all_stopwords
tokenize = lambda text: [token.group() for token in RE_WORD.finditer(text.lower()) if token not in all_stopwords]

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer

doc_locks = {docid: threading.Lock() for docid in doc_lengths}



#### old title search
def search_query_title_bm25(query):
  stemmer = PorterStemmer()
  query_weights = {}
  doc_scores = defaultdict(float)
  query = tokenize(query.lower())
  query = [stemmer.stem(token) for token in query]

  query_length = len(query)

  # calculate weights for query terms
  for word in query:
    # if word is not in index, pay no attention to it
    if title_index.df.get(word, None) == None:
      continue
    else:
        if query_length == 1:
          tf = query.count(word) #/ query_length
          query_weights[word] = tf
        elif idf_scores_title[word] >= 1.8:
          tf = query.count(word) #/ query_length
          query_weights[word] = tf

  if len(query_weights.items()) == 0:
    ######## find way to calculate if word is not in index
    return []

  k1 = k3 = 1.2
  b = 0.75
  avg_doc_length = 2.6211557574449786  ###### avg doc length
  docid_B = defaultdict(float)

  #   for word in query_weights.keys():
  def process_word(word):
    pl = title_index.read_a_posting_list(base_dir='', w=word, bucket_name='ruby_bucket_327064358')
    tf_q = query_weights[word]

    for docid, tf in pl:
      if docid in docid_B:
        B = docid_B[docid]
      else:
        B = (1 - b + b * (title_lengths[docid] / avg_doc_length))
      tf_doc = tf #/ title_lengths[docid]
      with doc_locks[docid]:
        doc_scores[docid] += idf_scores_title[word] * (((k1 + 1) * tf_doc) / (tf_doc + B * k1)) * (((k3 + 1) * tf_q) / (k3 + tf_q))

  with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit tasks for each word in query_weights.keys()
    for word in query_weights.keys():
      executor.submit(process_word, word)

  ret = sorted([(doc_id, score) for doc_id, score in doc_scores.items()], key=lambda x: x[1], reverse=True)[:100]
  ret = sorted(list(map(lambda x: (x[0], x[1] + 2*math.log(page_rank[x[0]], 10)), ret)), key=lambda x: x[1], reverse=True)
  # ret = list(map(lambda x: (x[0], doc_title_pairs[x[0]]), ret))
  return ret

##### old body search
def search_query_body_bm25(query):
  stemmer = PorterStemmer()
  query_weights = {}
  doc_scores = defaultdict(float)
  query = tokenize(query.lower())
  query = [stemmer.stem(token) for token in query]

  query_length = len(query)

  # calculate weights for query terms
  for word in query:
    # if word is not in index, pay no attention to it
    if body_index.df.get(word, None) == None:
      continue
    else:
        if query_length == 1:
          tf = query.count(word) #/ query_length
          query_weights[word] = tf
        elif idf_scores_body[word] >= 1.4:
          tf = query.count(word) #/ query_length
          query_weights[word] = tf

  if len(query_weights.items()) == 0:
    ######## find way to calculate if word is not in index
    return []

  k1 = k3 = 1.2
  b = 0.75
  avg_doc_length = 431.1623426698441  ###### avg doc length
  docid_B = defaultdict(float)

  #   for word in query_weights.keys():
  def process_word(word):
    pl = body_index.read_a_posting_list(base_dir='', w=word, bucket_name='ruby_bucket_327064358')
    tf_q = query_weights[word]

    for docid, tf in pl:
      if docid in docid_B:
        B = docid_B[docid]
      else:
        B = (1 - b + b * (doc_lengths[docid] / avg_doc_length))
      tf_doc = tf #/ doc_lengths[docid]
      with doc_locks[docid]:
        doc_scores[docid] += idf_scores_body[word] * (((k1 + 1) * tf_doc) / (tf_doc + B * k1)) * (((k3 + 1) * tf_q) / (k3 + tf_q))

  with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit tasks for each word in query_weights.keys()
    for word in query_weights.keys():
      executor.submit(process_word, word)

  ret = sorted([(doc_id, score) for doc_id, score in doc_scores.items()], key=lambda x: x[1], reverse=True)[:100]
  ret = sorted(list(map(lambda x: (x[0], x[1] + math.log(page_rank[x[0]], 10)), ret)), key=lambda x: x[1], reverse=True)
  # ret = list(map(lambda x: (x[0], doc_title_pairs[x[0]]), ret))
  return ret


# tf_idf normalization factor sum function

def calc_page_views(id):
  views = page_views[id]
  if views == 0:
    return 0
  else:
    return math.log(views, 10)

def sum_tfidf_scores(word_list):
  square = lambda x: x**2
  tot = 0
  for _, num in word_list:
    tot += square(num)
  return math.sqrt(tot)


### with word2vec
#
# def search_query_title_bm25(query):
#   stemmer = PorterStemmer()
#   query_weights = {}
#   doc_scores = defaultdict(float)
#   query = tokenize(query.lower())
#
#   # newq = query[:]
#   # for word in query:
#   #   try:
#   #     closest = model.most_similar(positive=[word], topn=2)
#   #     for synonym in closest:
#   #       newq.append(synonym[0])
#   #   except Exception:
#   #     continue
#
#   newq = [stemmer.stem(token) for token in query]
#   query_length = len(newq)
#
#   # calculate weights for query terms
#   for word in newq:
#
#     # if word is not in index, pay no attention to it
#     if title_index.df.get(word, None) == None:
#       continue
#     else:
#       if query_length == 1:
#         tf = newq.count(word)  # / query_length
#         query_weights[word] = tf
#       elif idf_scores_title[word] >= 2:
#         tf = newq.count(word)  # / query_length
#         query_weights[word] = tf
#
#   if len(query_weights.items()) == 0:
#     ######## find way to calculate if word is not in index
#     return []
#
#   k1 = k3 = 1.2
#   b = 0.75
#   avg_doc_length = 2.6211557574449786  ###### avg doc length
#   docid_B = defaultdict(float)
#
#   #   for word in query_weights.keys():
#   def process_word(word):
#     pl = title_index.read_a_posting_list(base_dir='', w=word, bucket_name='ruby_bucket_327064358')
#     tf_q = query_weights[word]
#
#     for docid, tf in pl:
#       if docid in docid_B:
#         B = docid_B[docid]
#       else:
#         B = (1 - b + b * (title_lengths[docid] / avg_doc_length))
#       tf_doc = tf / title_lengths[docid]
#       with doc_locks[docid]:
#         doc_scores[docid] += idf_scores_title[word] * (((k1 + 1) * tf_doc) / (tf_doc + B * k1)) * (
#                 ((k3 + 1) * tf_q) / (k3 + tf_q))
#
#   with concurrent.futures.ThreadPoolExecutor() as executor:
#     # Submit tasks for each word in query_weights.keys()
#     for word in query_weights.keys():
#       executor.submit(process_word, word)
#
#   ret = sorted([(doc_id, score) for doc_id, score in doc_scores.items()], key=lambda x: x[1], reverse=True)[:100]
#   #ret = sorted(list(map(lambda x: (x[0], x[1] + 2 * math.log(page_rank[x[0]], 10)), ret)), key=lambda x: x[1],reverse=True)
#   # ret = list(map(lambda x: (x[0], doc_title_pairs[x[0]]), ret))
#   return ret
#
#
# def search_query_body_bm25(query):
#   # stemmer = PorterStemmer()
#   query_weights = {}
#   doc_scores = defaultdict(float)
#   query = tokenize(query.lower())
#
#   query = list(filter(lambda x: x in body_index.df, query))
#   newq = query[:]
#   if len(query) == 1:
#     # If query has only one word, get the most similar words to that word
#     try:
#       closest = model.most_similar(positive=query, topn=5)
#       for synonym in closest:
#         newq.append(synonym[0])
#     except Exception:
#       pass
#   else:
#     # If query has more than one word, calculate closest for every pair of consecutive words
#     for i in range(len(query) - 1):
#       pair = query[i:i + 2]  # Get consecutive pair of words
#       try:
#         closest = model.most_similar(positive=pair, topn=3)
#         for synonym in closest:
#           newq.append(synonym[0])
#       except Exception:
#         pass
#
#
#   # newq = query[:]
#   # for word in query:
#   #   try:
#   #     closest = model.most_similar(positive=[word], topn=3)
#   #     for synonym in closest:
#   #       newq.append(synonym[0])
#   #   except Exception:
#   #     continue
#
#   # newq = [stemmer.stem(token) for token in newq]
#   query_length = len(newq)
#
#   # calculate weights for query terms
#   for word in newq:
#     # if word is not in index, pay no attention to it
#     if body_index.df.get(word, None) == None:
#       continue
#     else:
#       if query_length == 1:
#         tf = newq.count(word)  # / query_length
#         query_weights[word] = tf
#       elif idf_scores_body[word] >= 1.5:
#         tf = newq.count(word)  # / query_length
#         query_weights[word] = tf
#
#     # return query_weights.items()
#   if len(query_weights.items()) == 0:
#     ######## find way to calculate if word is not in index
#     return []
#   k1 = k3 = 1.2
#   b = 0.75
#   avg_doc_length = 431.1623426698441  ###### avg doc length
#   docid_B = defaultdict(float)
#
#   #   for word in query_weights.keys():
#   def process_word(word):
#     pl = body_index.read_a_posting_list(base_dir='postings_gcp', w=word, bucket_name='ruby_bucket_327064358')
#     tf_q = query_weights[word]
#
#     for docid, tf in pl:
#       if docid in docid_B:
#         B = docid_B[docid]
#       else:
#         B = (1 - b + b * (doc_lengths[docid] / avg_doc_length))
#       tf_doc = tf  # / doc_lengths[docid]
#       with doc_locks[docid]:
#         doc_scores[docid] += idf_scores_body[word] * (((k1 + 1) * tf_doc) / (tf_doc + B * k1)) * (
#                 ((k3 + 1) * tf_q) / (k3 + tf_q))
#
#   with concurrent.futures.ThreadPoolExecutor() as executor:
#     # Submit tasks for each word in query_weights.keys()
#     for word in query_weights.keys():
#       executor.submit(process_word, word)
#
#   ret = sorted([(doc_id, score) for doc_id, score in doc_scores.items()], key=lambda x: x[1], reverse=True)[:100]
#   #ret = sorted(list(map(lambda x: (x[0], x[1] + 2 * math.log(page_rank[x[0]], 10)), ret)), key=lambda x: x[1],reverse=True)
#   # ret = list(map(lambda x: (x[0], doc_title_pairs[x[0]]), ret))
#   return ret

def search(query):
  query = query.lower()
  doc_scores = defaultdict(int)
  body_scores = 0
  title_scores = 0

  query_len = len(tokenize(query.lower()))


  def update_dic(scores, weight):
    for id, score in scores:
      if id in doc_scores:
        doc_scores[id] += weight * score
      else:
        doc_scores[id] = weight * score

  # Function to run search_query_body in a separate thread
  def run_search_query_body():
    nonlocal body_scores
    body_scores = search_query_body_bm25(query)

    # Function to run search_query_title in a separate thread

  def run_search_query_title():
    nonlocal title_scores
    title_scores = search_query_title_bm25(query)

  # Create and start threads for search_query_body and search_query_title
  thread_body = threading.Thread(target=run_search_query_body)
  thread_title = threading.Thread(target=run_search_query_title)
  thread_body.start()
  thread_title.start()

  # Wait for both threads to finish
  thread_body.join()
  thread_title.join()

  if query_len <= 2:
    update_dic(body_scores, weight=0)
    update_dic(title_scores, weight=1)
  elif query_len >= 5:
    update_dic(body_scores, weight=4)
    update_dic(title_scores, weight=1)
  else:
    update_dic(body_scores, weight=1)
    update_dic(title_scores, weight=6)


  ret = sorted([(doc_id, score) for doc_id, score in doc_scores.items()], key=lambda x: x[1], reverse=True)[:100]
  ret = sorted(list(map(lambda x: (x[0], x[1] + 5*math.log(page_rank[x[0]],2) + calc_page_views(x[0])), ret)), key=lambda x: x[1], reverse=True)
  ret = list(map(lambda x: (str(x[0]), doc_title_pairs[x[0]]), ret))
  return ret