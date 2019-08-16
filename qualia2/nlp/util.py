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

def analogy(a, b, c, word2idx, wordvec, n=5):
    ''' analogy\n
    Predicts word relationship like a:b = c:?
    Args:
        a (str): input string
        b (str): input string
        c (str): input string
        word2idx (dict): word to index map
        wordvec (Embedding): ector representation of words
        n (int): top n similar words to show
    '''
    assert a in word2idx
    assert b in word2idx
    assert c in word2idx

    print('[*] analogy {}:{} = {}:?'.format(a,b,c))
    idx2word = {v: k for k, v in word2idx.items()}
    a_vec, b_vec, c_vec = wordvec(word2idx[a]), wordvec(word2idx[b]), wordvec(word2idx[c])
    query_vec = b_vec - a_vec + c_vec
    
    similarity = np.array([float(cosine_similarity(wordvec(i), query_vec).data) for i in range(len(idx2word))])
    idx = similarity.argsort()[::-1]
    for i in range(n):
        print('{}: {}'.format(idx2word[int(idx[i])], similarity[int(idx[i])]))
