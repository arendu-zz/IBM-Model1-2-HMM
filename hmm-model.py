__author__ = 'arenduchintala'
#import _numpypy.multiarray as np
import numpy as np
import logutils as lu
from math import log, exp
from pprint import pprint
import pdb, sys, codecs
from pprint import pprint as pp

np.set_printoptions(precision=4, linewidth=180)

BOUNDRY_STATE = "###"
alignment_probs = {}
translations_probs = {}


def get_possible_states(target_seq, source_seq):
    possible_states = []
    target_start_idx = [i for i, x in enumerate(target_seq) if x == BOUNDRY_STATE]
    source_start_idx = [i for i, x in enumerate(source_seq) if x == BOUNDRY_STATE]
    sent_count = -1
    for idx, target_token in enumerate(target_seq):
        if target_token == BOUNDRY_STATE:
            possible_states.append([(BOUNDRY_STATE, BOUNDRY_STATE)])
            sent_count += 1
        else:
            try:
                et = target_start_idx[sent_count + 1]
                es = source_start_idx[sent_count + 1]
            except IndexError:
                et = len(target_seq)
                es = len(source_seq)

            #current_target_sentence = target_seq[target_start_idx[sent_count] + 1:et]
            current_source_sentence = source_seq[source_start_idx[sent_count] + 1:es]
            ps = [(i, st) for i, st in enumerate(current_source_sentence)]
            possible_states.append(ps)
    return possible_states


def get_transition(current_state, prev_state):
    return alignment_probs.get((current_state, prev_state), float('-inf'))


def get_emission(obs, state):
    return translations_probs.get((obs, state), float('-inf'))


def do_accumilate_posterior_obs(accumilation_dict, obs, aj, ei, posterior_unigram_val):
    # these are actual counts in log space!!
    if isinstance(obs, str) and (not isinstance(aj, tuple)) and isinstance(ei, str):
        if ('count_obs', obs) in accumilation_dict:
            accumilation_dict[('count_obs', obs)] = lu.logadd(accumilation_dict[('count_obs', obs)], posterior_unigram_val)
        else:
            accumilation_dict[('count_obs', obs)] = posterior_unigram_val
        if ('count_state', aj) in accumilation_dict:
            accumilation_dict[('count_state', aj)] = lu.logadd(accumilation_dict[('count_state', aj)], posterior_unigram_val)
        else:
            accumilation_dict[('count_state', aj)] = posterior_unigram_val

        if ('count_emission', obs, ei) in accumilation_dict:
            accumilation_dict[('count_emission', obs, ei)] = lu.logadd(accumilation_dict[('count_emission', obs, ei)],
                                                                       posterior_unigram_val)
        else:
            accumilation_dict[('count_emission', obs, ei)] = posterior_unigram_val
            # doing total counts ...
        if ('any_emission_from', ei) in accumilation_dict:
            accumilation_dict[('any_emission_from', ei)] = lu.logadd(accumilation_dict[('any_emission_from', ei)], posterior_unigram_val)
        else:
            accumilation_dict[('any_emission_from', ei)] = posterior_unigram_val
        return accumilation_dict
    else:
        print 'obs must be string, aj must be str, ei must be string'
        exit()


def do_accumilate_posterior_bigrams(accumilation_dict, aj, aj_1, posterior_bigram_val):
    # these are actual counts in log space!!
    if not isinstance(aj, tuple) or isinstance(aj_1, tuple):
        if ('count_transition', aj, aj_1) not in accumilation_dict:
            accumilation_dict[('count_transition', aj, aj_1)] = posterior_bigram_val
        else:
            accumilation_dict[('count_transition', aj, aj_1)] = lu.logadd(accumilation_dict[('count_transition', aj, aj_1)],
                                                                          posterior_bigram_val)

        if ('any_transition_from', aj_1) not in accumilation_dict:
            accumilation_dict[('any_transition_from', aj_1)] = posterior_bigram_val
        else:
            accumilation_dict[('any_transition_from', aj_1)] = lu.logadd(accumilation_dict[('any_transition_from', aj_1)],
                                                                         posterior_bigram_val)
        return accumilation_dict
    else:
        print 'aj and aj_1 should be str ### or int', aj, aj_1
        exit()


def do_append_posterior_unigrams(appending_dict, position, state, posterior_unigram_val):
    if position in appending_dict:
        appending_dict[position].append((state, posterior_unigram_val))
    else:
        appending_dict[position] = [(state, posterior_unigram_val)]
    return appending_dict


def flatten_backpointers(bt):
    reverse_bt = []
    while len(bt) > 0:
        x = bt.pop()
        reverse_bt.append(x)
        if len(bt) > 0:
            bt = bt.pop()
    reverse_bt.reverse()
    return reverse_bt


def get_viterbi_and_forward(obs_sequence, trelis):
    pi = {(0, (BOUNDRY_STATE, BOUNDRY_STATE)): 0.0}
    alpha_pi = {(0, (BOUNDRY_STATE, BOUNDRY_STATE)): 0.0}
    #pi[(0, START_STATE)] = 1.0  # 0,START_STATE
    arg_pi = {(0, (BOUNDRY_STATE, BOUNDRY_STATE)): []}
    for k in range(1, len(obs_sequence)):  # the words are numbered from 1 to n, 0 is special start character
        for v in trelis[k]:  # [1]:
            max_prob_to_bt = {}
            sum_prob_to_bt = []
            target_token = obs_sequence[k]
            source_token = v[1]
            for u in trelis[k - 1]:  # [1]:
                aj = v[0]
                aj_1 = u[0]
                q = get_transition(aj, aj_1)
                e = get_emission(target_token, source_token)
                #print k
                #print v, '|', u
                #print aj, '|', aj_1, '=', q
                #print target_token, '|', source_token, '=', e
                p = pi[(k - 1, u)] + q + e
                alpha_p = alpha_pi[(k - 1, u)] + q + e
                if len(arg_pi[(k - 1, u)]) == 0:
                    bt = [u]
                else:
                    bt = [arg_pi[(k - 1, u)], u]
                max_prob_to_bt[p] = bt
                sum_prob_to_bt.append(alpha_p)

            max_bt = max_prob_to_bt[max(max_prob_to_bt)]
            new_pi_key = (k, v)
            pi[new_pi_key] = max(max_prob_to_bt)
            #print 'mu   ', new_pi_key, '=', pi[new_pi_key], exp(pi[new_pi_key])
            alpha_pi[new_pi_key] = lu.logadd_of_list(sum_prob_to_bt)
            #print 'alpha', new_pi_key, '=', alpha_pi[new_pi_key], exp(alpha_pi[new_pi_key])
            arg_pi[new_pi_key] = max_bt

    max_bt = max_prob_to_bt[max(max_prob_to_bt)]
    max_p = max(max_prob_to_bt)
    max_bt = flatten_backpointers(max_bt)
    return max_bt, max_p, alpha_pi


def get_backwards(obs, trelis, alpha_pi):
    n = len(obs) - 1  # index of last word
    beta_pi = {(n, (BOUNDRY_STATE, BOUNDRY_STATE)): 0.0}
    S = alpha_pi[(n, (BOUNDRY_STATE, BOUNDRY_STATE))]  # from line 13 in pseudo code
    posterior_unigrams = {}
    posterior_obs_accumilation = {}
    posterior_bigrams_accumilation = {}
    for k in range(n, 0, -1):
        for v in trelis[k]:
            pb = beta_pi[(k, v)]
            aj = v[0]
            source_token = v[1]
            posterior_unigram_val = beta_pi[(k, v)] + alpha_pi[(k, v)] - S
            posterior_obs_accumilation = do_accumilate_posterior_obs(posterior_obs_accumilation, obs[k], aj, source_token,
                                                                     posterior_unigram_val)
            posterior_unigrams = do_append_posterior_unigrams(posterior_unigrams, k, v, posterior_unigram_val)

            for u in trelis[k - 1]:
                #print 'reverse transition', 'k', k, 'u', u, '->', 'v', v
                aj_1 = u[0]
                q = get_transition(aj, aj_1)
                target_token = obs[k]
                e = get_emission(target_token, source_token)
                p = q + e
                beta_p = pb + p
                new_pi_key = (k - 1, u)
                if new_pi_key not in beta_pi:  # implements lines 16
                    beta_pi[new_pi_key] = beta_p
                else:
                    beta_pi[new_pi_key] = lu.logadd(beta_pi[new_pi_key], beta_p)
                    #print 'beta     ', new_pi_key, '=', beta_pi[new_pi_key], exp(beta_pi[new_pi_key])
                posterior_bigram_val = alpha_pi[(k - 1, u)] + p + beta_pi[(k, v)] - S
                #posterior_bigram_val = "%.3f" % (exp(alpha_pi[(k - 1, u)] + p + beta_pi[(k, v)] - S))
                posterior_bigrams_accumilation = do_accumilate_posterior_bigrams(posterior_bigrams_accumilation, aj, aj_1,
                                                                                 posterior_bigram_val)
    return posterior_unigrams, posterior_bigrams_accumilation, posterior_obs_accumilation, S, beta_pi


def format_alignments(init_aligns):
    aligns = {}
    for line in init_aligns:
        [snum, inum, jnum] = line.split()
        s_index = int(snum)
        a = aligns.get(snum, [BOUNDRY_STATE])
        j_index = int(jnum)
        i_index = int(inum)
        if len(a) - 1 < j_index:
            pad = [0] * (j_index - (len(a) - 1))
            a = a + pad
            a[j_index] = i_index
        else:
            a[j_index] = i_index
        aligns[snum] = a
    A = []
    for k, a in sorted(aligns.iteritems()):
        A = A + a
    A.append(BOUNDRY_STATE)
    return A


def get_alignment_mle(init_alignments):
    alignment_mles = {}
    num_alignment_states = len([i for i in init_alignments if i != BOUNDRY_STATE])
    alignment_bigrams = [(init_alignments[i], init_alignments[i - 1]) for i in range(1, len(init_alignments))]
    alignments_count = {}
    for ab in alignment_bigrams:
        alignments_count[ab[1]] = alignments_count.get(ab[1], 0) + 1
        alignments_count[ab] = alignments_count.get(ab, 0) + 1

    for ap in alignments_count:
        if isinstance(ap, tuple):
            #its a bigram
            (aj, aj_1) = ap
            alignment_mles[ap] = log(float(alignments_count[ap]) / float(alignments_count[aj_1]))
        else:
            #its a unigram
            alignment_mles[ap] = log(float(alignments_count[ap]) / float(num_alignment_states))

    #print alignment_bigrams
    #print alignments_count
    #print alignments_probs
    return alignment_mles


def update_alignment_mle(posterior_alignment_counts):
    for k in posterior_alignment_counts:
        if k[0] == 'count_transition':
            (comment, aj, aj_1) = k
            count_ajaj_1 = posterior_alignment_counts['count_transition', aj, aj_1]
            count_aj_1 = posterior_alignment_counts['any_transition_from', aj_1]
            if count_aj_1 == float('-inf'):
                alignment_probs[aj, aj_1] = float('-inf')
            else:
                alignment_probs[aj, aj_1] = count_ajaj_1 - count_aj_1
        else:
            (comment, aj_1) = k
            count_aj_1 = posterior_alignment_counts['any_transition_from', aj_1]
            alignment_probs[aj_1] = count_aj_1


def update_translation_mle(posterior_emission_counts):
    for f, e in translations_probs:
        counts_fe = posterior_emission_counts['count_emission', f, e]
        count_e = posterior_emission_counts['any_emission_from', e]
        if count_e == float('-inf'):
            translations_probs[f, e] = float('-inf')
        else:
            translations_probs[f, e] = counts_fe - count_e


def get_translation_mle(init_trans):
    translations_mle = {}
    for line in init_trans:
        [fi, ej, p] = line.split()
        translations_mle[fi, ej] = log(float(p))
    translations_mle[BOUNDRY_STATE, BOUNDRY_STATE] = 0.0
    return translations_mle


def parseargs(args):
    try:
        source_idx = args.index('-s')
        target_idx = args.index('-t')
        source = args[source_idx + 1]
        target = args[target_idx + 1]
        init_translations = args[args.index('-it') + 1]
        init_align = args[args.index('-ia') + 1]
        save_translations_learned = args[args.index('-p') + 1]
        save_alignment_out = args[args.index('-a') + 1]
        source_alignment_test = args[args.index('-as') + 1]
        target_alignment_test = args[args.index('-at') + 1]
        return source, target, init_translations, init_align, save_translations_learned, save_alignment_out, source_alignment_test, target_alignment_test
    except (ValueError, IndexError) as er:
        print 'Usage: python model1.py -t [train target] -s [train source] -it [initial translations]' \
              ' -p [save translations] ' \
              '-a [save alignment test] -as [alignment test source] -at [alignment test target]'
        #return 'dummy.en', 'dummy.es', 'dummy.trans', 'dummy.align', 's', 's', 's', 's'
        return 'corpus.en', 'corpus.es', 'model1-fwd-out.trans', 'out.ali', 'hmm.trans', 'hmm.align', 'corpus.en', 'corpus.es'
        #exit()


if __name__ == "__main__":

    source, target, init_translations, init_alignments, save_translations_learned, save_alignment_out, source_alignment_test, target_alignment_test = parseargs(
        sys.argv)

    corpus_source = open(source, 'r').readlines()
    corpus_target = open(target, 'r').readlines()
    init_alignments = format_alignments(open(init_alignments, 'r').readlines())
    init_translations = open(init_translations, 'r').readlines()
    joined_source = (BOUNDRY_STATE + ' NULL ').join(corpus_source)
    joined_target = (BOUNDRY_STATE + ' ').join(corpus_target)
    source_tokens = joined_source.split()
    target_tokens = joined_target.split()
    source_tokens.insert(0, 'NULL')
    source_tokens.insert(0, BOUNDRY_STATE)
    source_tokens.append(BOUNDRY_STATE)
    target_tokens.insert(0, BOUNDRY_STATE)
    target_tokens.append(BOUNDRY_STATE)
    '''
    print source_tokens
    print init_alignments
    print target_tokens
    print alignment_probs
    print translations_probs
    '''
    alignment_probs = get_alignment_mle(init_alignments)
    translations_probs = get_translation_mle(init_translations)
    trelis = get_possible_states(target_tokens, source_tokens)
    #for obs, ps in zip(target_tokens, trelis):
    #    print obs, '<--', ps
    for i in range(5):
        #pdb.set_trace()
        max_bt, max_p, alpha_pi = get_viterbi_and_forward(target_tokens, trelis)
        posterior_unigrams, posterior_bigrams_accumilation, posterior_obs_accumilation, S, beta_pi = get_backwards(target_tokens, trelis,
                                                                                                                   alpha_pi)
        update_alignment_mle(posterior_bigrams_accumilation)
        update_translation_mle(posterior_obs_accumilation)
        print 'iteration', i, max_p, S






