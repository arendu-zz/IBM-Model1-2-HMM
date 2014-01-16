__author__ = 'arenduchintala'

"""
parameters in model 1:
delta[k,i,j] = translation[ foreign[k,i], english[k,j]] / sum_j_0toL (translation( foreign[k,i], english[k,j]))
k = kth sentences in the corpus
i = ith word in the kth sentence in the foreign corpus
j = jth word in the kth sentence in the english corpus
L = total number of words in the kth sentence in the english corpus
M = total number of words in the kth sentence in the foreign corpus
"""
"""
counts in model 1:
count[ejk, fik] = count[ejk, fik] + delta[k,i,j]
count[ejk] = count[ejk] + delta[k,i,j]
"""
"""
translation
translation[f,e] = c(f,e) / c(e)
"""
"""
for debugging purposes
https://class.coursera.org/nlangp-001/forum/thread?thread_id=940#post-4052
"""
import numpy as np
import pdb, sys, codecs
from pprint import pprint as pp

np.set_printoptions(precision=4, linewidth=180)


def display_best_alignment(ak, en, es):
    lk = len(en)
    mk = len(es)
    k_mat = np.zeros((mk, lk))
    for jk in range(lk):
        for ik in range(mk):
            k_mat[ik][jk] = delta[ak, ik, jk]
    print ' '.join(en)
    print ' '.join(es)
    for ik, max_jk in enumerate(np.argmax(k_mat, 1)):
        print ik, max_jk, corpus_es[ak][ik], corpus_en[ak][max_jk]


delta = {}
translations = {}
counts = {}
corpus_en = open('corpus.en', 'r').readlines()
corpus_es = open('corpus.es', 'r').readlines()
initial_translation = {}

"""
initialization
"""
for k, sentence_en in enumerate(corpus_en):
    sentence_es = corpus_es[k]
    tokens_en = sentence_en.split()
    tokens_en.insert(0, 'NULL')
    tokens_es = sentence_es.split()
    corpus_en[k] = tokens_en
    corpus_es[k] = tokens_es
    for e in tokens_en:
        n_e = initial_translation.get(e, set())
        n_e.update(tokens_es)
        initial_translation[e] = n_e
#print 'initial n:'
#pp(initial_translation)
for k, v in initial_translation.iteritems():
    for v_es in v:
        translations[v_es, k] = 1.0 / len(v)
#print 'initial t:'
#pp(translations)
"""
EM iterations
"""

for iter in range(5):
    counts = dict.fromkeys(counts.iterkeys(), 0.0)
    for k, tokens_en in enumerate(corpus_en):
        #print iter, k, len(delta), len(translations)
        sys.stdout.write('iteration: %d sentence %d len delta %d len translations %d\r' % (iter, k, len(delta), len(translations)))
        sys.stdout.flush()
        tokens_es = corpus_es[k]
        t_mat = np.zeros((len(tokens_es), len(tokens_en)))
        #print t_mat, t_mat.shape
        for j in range(0, len(tokens_en)):
            for i in range(0, len(tokens_es)):
                t_mat[i][j] = translations[tokens_es[i], tokens_en[j]]
        t_sum = np.sum(t_mat, 1)
        #print t_mat
        #print t_sum
        for j in range(0, len(tokens_en)):
            for i in range(0, len(tokens_es)):
                delta[k, i, j] = t_mat[i][j] / t_sum[i]
                counts[tokens_es[i], tokens_en[j]] = counts.get((tokens_es[i], tokens_en[j]), 0.0) + delta[k, i, j]
                counts[tokens_en[j]] = counts.get(tokens_en[j], 0.0) + delta[k, i, j]
                #print tokens_es[i], tokens_en[j], counts[tokens_es[i], tokens_en[j]]
                #print tokens_en[j], counts[tokens_en[j]]
                #print 'iteration:', iter, 'sentence', k
    """
    update translations
    """
    for t_es_i, t_en_j in translations:
        translations[t_es_i, t_en_j] = counts[t_es_i, t_en_j] / counts[t_en_j]

    """
    print 'iter', iter
    print 'delta:'
    pp(delta)
    print 'counts:'
    pp(counts)
    print 'translations:'
    pp(translations)
    """
    """
    check how the alignment looks for a particular training pair, a particular sentence
    """

    """display_best_alignment(1012, corpus_en[1012], corpus_es[1012])
    display_best_alignment(829, corpus_en[829], corpus_es[829])
    display_best_alignment(2204, corpus_en[2204], corpus_es[2204])
    display_best_alignment(4942, corpus_en[4942], corpus_es[4942])"""
writer = open('translations.txt', 'w')
for k, v in translations.iteritems():
    writer.write(str(' '.join(k)) + '\t' + str(v) + '\n')
writer.flush()
writer.close()
"""
writer = open('alignment_test.p1.out', 'w')

dev_en = open('test.en', 'r').readlines()
dev_es = open('test.es', 'r').readlines()
for dk in range(len(dev_en)):
    tokens_en = dev_en[dk].split()
    tokens_en.insert(0, 'NULL')
    tokens_es = dev_es[dk].split()
    for i, token_es in enumerate(tokens_es):
        max_p = 0.0
        max_j = 0.0
        for j, token_en in enumerate(tokens_en):
            if translations[token_es, token_en] > max_p:
                max_p = translations[token_es, token_en]
                max_j = j
        if max_j > 0:
            writer.write(str(dk + 1) + ' ' + str(max_j) + ' ' + str(i + 1) + '\n')
writer.flush()
writer.close()
"""
