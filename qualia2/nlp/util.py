# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import Tensor
from ..functions import cosine_similarity

def most_similar(query, word2idx, wordvecs, n=5):
    ''' most_similar\n
    Look up most similar words in the embedding
    Args:
        query (str): query text
        word2idx (dict): word to index map
        wordvecs (Embedding): vector representation of words
        n (int): top n similar words to show
    '''
    if query not in word2idx:
        raise Exception('[*] \'{}\' is unknown.'.format(query))
    print('[*] query: ' + query)
    idx2word = {v: k for k, v in word2idx.items()}
    query_vec = wordvecs(word2idx[query])

    similarity = np.array([float(cosine_similarity(wordvecs(i), query_vec).data) for i in range(len(idx2word))])
    idx = similarity.argsort()[::-1]
    for i in range(n+1):
        if idx2word[int(idx[i])] == query:
            continue
        print('{}: {}'.format(idx2word[int(idx[i])], similarity[int(idx[i])]))
