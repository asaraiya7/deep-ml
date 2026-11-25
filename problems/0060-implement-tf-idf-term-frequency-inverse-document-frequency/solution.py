import numpy as np
from collections import Counter

def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents.

    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document
    """
    idf = []
    doc_counter_list = [Counter(doc) for doc in corpus]

    for query_x in query:
        # (1) Find no. of docs which contains query word
        doc_freq = 0
        for doc_counter in doc_counter_list:
            if query_x in doc_counter:
                doc_freq+= 1
                    
        # (2) Compute smoothed IDF
        smoothed_idf = float(np.log((len(corpus) + 1) / (doc_freq + 1))) + 1

        # Find idf score for given query for all docs 
        per_doc_scores = []
        for idx, doc in enumerate(corpus):
            #(3): Compute term frequency 
            tf_d = doc_counter_list[idx][query_x] / len(doc)
            doc_idf_score = tf_d * smoothed_idf
            per_doc_scores.append(float(np.round(doc_idf_score, 5)))
        
        idf.append(per_doc_scores)
    
    return np.array(idf).T.tolist()
            