__author__ = 'arenduchintala'


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
q = {}
translations = {}
counts = {}
corpus_en = open('corpus.en', 'r').readlines()
corpus_es = open('corpus.es', 'r').readlines()

"""
initialization
"""
for line in open('translations.txt', 'r').readlines():
    [fi, ej, p] = line.split()
    translations[fi, ej] = float(p)

#print 'initial t:'
#pp(translations)

for k, sentence_en in enumerate(corpus_en):
    sentence_es = corpus_es[k]
    tokens_en = sentence_en.split()
    tokens_en.insert(0, 'NULL')
    tokens_es = sentence_es.split()
    corpus_en[k] = tokens_en
    corpus_es[k] = tokens_es
    mk = len(tokens_es)
    lk = len(tokens_en)
    for e, ej in enumerate(tokens_en):
        for f, fi in enumerate(tokens_es):
            q[e, f, lk, mk] = 1.0 / lk

#print 'initial q:'
#pp(q)
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
        mk = len(tokens_es)
        lk = len(tokens_en)
        qt_mat = np.zeros((mk, lk))
        #print t_mat, t_mat.shape
        for j in range(0, lk):
            for i in range(0, mk):
                qt_mat[i][j] = q[j, i, lk, mk] * translations[tokens_es[i], tokens_en[j]]
        qt_sum = np.sum(qt_mat, 1)
        #print qt_mat, qt_sum
        for j in range(0, lk):
            for i in range(0, mk):
                delta[k, i, j] = qt_mat[i][j] / qt_sum[i]
                counts[tokens_es[i], tokens_en[j]] = counts.get((tokens_es[i], tokens_en[j]), 0.0) + delta[k, i, j]
                counts[tokens_en[j]] = counts.get(tokens_en[j], 0.0) + delta[k, i, j]
                counts[j, i, lk, mk] = counts.get((j, i, lk, mk), 0.0) + delta[k, i, j]
                counts[i, lk, mk] = counts.get((i, lk, mk), 0.0) + delta[k, i, j]

    """
    update translations
    """
    for t_es_i, t_en_j in translations:
        translations[t_es_i, t_en_j] = counts[t_es_i, t_en_j] / counts[t_en_j]
    for qj, qi, ql, qm in q:
        q[qj, qi, ql, qm] = counts[qj, qi, ql, qm] / counts[qi, ql, qm]
    """
    print 'iter', iter
    print 'delta:'
    pp(delta)
    print 'counts:'
    pp(counts)
    print 'translations:'
    pp(translations)
    print 'q:'
    pp(q)
    """

    """
    check how the alignment looks for a particular training pair, a particular sentence
    """

    """display_best_alignment(1012, corpus_en[1012], corpus_es[1012])
    display_best_alignment(829, corpus_en[829], corpus_es[829])
    display_best_alignment(2204, corpus_en[2204], corpus_es[2204])
    display_best_alignment(4942, corpus_en[4942], corpus_es[4942])"""

writer = open('alignment_test.p2.out', 'w')

dev_en = open('test.en', 'r').readlines()
dev_es = open('test.es', 'r').readlines()
for dk in range(len(dev_en)):
    tokens_en = dev_en[dk].split()
    tokens_en.insert(0, 'NULL')
    tokens_es = dev_es[dk].split()
    l = len(tokens_en)
    m = len(tokens_es)
    for i, token_es in enumerate(tokens_es):
        max_p = 0.0
        max_j = 0.0
        for j, token_en in enumerate(tokens_en):
            if q[j, i, l, m] * translations[token_es, token_en] > max_p:
                max_p = q[j, i, l, m] * translations[token_es, token_en]
                max_j = j
        if max_j > 0:
            writer.write(str(dk + 1) + ' ' + str(max_j) + ' ' + str(i + 1) + '\n')
writer.flush()
writer.close()

