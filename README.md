# Search Engine for Wikipedia

As a project, we built a search engine on the entire Wikipedia corpus. 

This engine works with the BM25 similarity function and through various optimizations such as threading, stemming and query filtering.


This repository contains four main files:
1. inverted_index_gcp.py
   - this file is an implementation of an inverted index which supports the creation of large indices and the storing of posting lists on the disk.
2. search_frontend
   - the creation of a flask app that will connect to the backend and receive queries from the user.
3. backend_project
   - the implementation of the search engine. CAUTION: this file loads the indices and other large files into memory
4. build_indices
   - a python notebook with code to show how each index was built. NOTE: the building of these indices was done on Google Cloud Platform, and the calculations were performed through Spark.
   this notebook contains the code to build an index on the body text of wikipedia, the titles of wikipedia, the pagerank of wikipedia,
   and (though not used in the final engine) an import of a word2vec model trained on the wikipedia corpus.
   
