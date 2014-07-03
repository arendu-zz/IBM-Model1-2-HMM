__author__ = 'arenduchintala'

import logutils as lu
from math import log, fabs
import pdb, codecs, sys


BOUNDRY_STATE = "###"
COUNT_TYPE = 'count'
jump_counts = {}  # used to compute jump probabilities p(aj | aj-1, I)
translations_probs = {}  # used to compute translation probabilities p(f_i | e_j)


def jump_key(j1, j0, sent_len):
    if j1 == BOUNDRY_STATE:
        return COUNT_TYPE, fabs(sent_len - j0)
    elif j0 == BOUNDRY_STATE:
        return COUNT_TYPE, j1
    else:
        return COUNT_TYPE, fabs(j1 - j0)  # eq (5) assume non-negative jump widths!


def get_trellis(target_seq, source_seq):
    '''
    This method returns a trellis for each target-source sentence pair
    A trellis is just the possible hidden states (beam limited source tokens) for a observation (target token)
    '''
    t = []
    for it, f in enumerate(target_seq):
        if f == BOUNDRY_STATE:
            t.append([(BOUNDRY_STATE, BOUNDRY_STATE)])
        else:
            tups = [(translations_probs.get((f, e), float('-inf')), f, (i + 1, e)) for i, e in enumerate(
                source_seq[1:-1])]  # [1:-1] because we know that the boundry symbols dont emit any of the target tokens
            tups.sort(reverse=True)  # pick best candidates for translation
            tups = tups[:10]  # beam limited to 5, 10, 15 etc
            tups.append((translations_probs[f, 'NULL'], f, (it, 'NULL')))
            (ps, fs, ts) = zip(*tups)
            t.append(list(ts))
    return t


def get_jump_transition(current_state, prev_state, sent_length):
    '''
    This method implements eq (5) in the Vogel & Ney paper (HMM-Based Word Alignment in Statistical Translation)
    it returns the probability P(aj | aj-1, L)
    '''
    jkey = jump_key(current_state, prev_state, sent_length)
    if jkey in jump_counts:
        # TODO this demon computation can be reused!
        # Another TODO: using a normal distribution to get probability of jump widths might be much faster!
        denom = float('-inf')
        for l in range(sent_length):
            jl_key = jump_key(l, prev_state, sent_length)
            denom = lu.logadd(denom, jump_counts.get(jl_key, float('-inf')))
        return jump_counts[jkey] - denom
    else:
        return -100.00


def get_emission(obs, state):
    return translations_probs.get((obs, state), float('-inf'))


def do_accumilate_posterior_obs(accumilation_dict, obs, aj, ei, posterior_unigram_val):
    # these are actual counts in log space!!
    if isinstance(obs, basestring) and (not isinstance(aj, tuple)) and isinstance(ei, basestring):
        if ('count_obs', obs) in accumilation_dict:
            accumilation_dict[('count_obs', obs)] = lu.logadd(accumilation_dict[('count_obs', obs)],
                                                              posterior_unigram_val)
        else:
            accumilation_dict[('count_obs', obs)] = posterior_unigram_val
        if ('count_state', aj) in accumilation_dict:
            accumilation_dict[('count_state', aj)] = lu.logadd(accumilation_dict[('count_state', aj)],
                                                               posterior_unigram_val)
        else:
            accumilation_dict[('count_state', aj)] = posterior_unigram_val

        if ('count_emission', obs, ei) in accumilation_dict:
            accumilation_dict[('count_emission', obs, ei)] = lu.logadd(accumilation_dict[('count_emission', obs, ei)],
                                                                       posterior_unigram_val)
        else:
            accumilation_dict[('count_emission', obs, ei)] = posterior_unigram_val
            # doing total counts ...
        if ('any_emission_from', ei) in accumilation_dict:
            accumilation_dict[('any_emission_from', ei)] = lu.logadd(accumilation_dict[('any_emission_from', ei)],
                                                                     posterior_unigram_val)
        else:
            accumilation_dict[('any_emission_from', ei)] = posterior_unigram_val
        return accumilation_dict
    else:
        print 'obs must be string, aj must be str, ei must be string'
        exit()


def do_accumilate_posterior_bigrams_jump(accumilation_dict, aj, aj_1, posterior_bigram_val, sent_length):
    # these are actual counts in log space!!
    if not isinstance(aj, tuple) or isinstance(aj_1, tuple):
        jkey = jump_key(aj, aj_1, sent_length)
        accumilation_dict[jkey] = lu.logadd(accumilation_dict.get(jkey, float('-inf')), posterior_bigram_val)
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


def get_viterbi_and_forward(obs_sequence, trelis, source_len):
    pi = {(0, (BOUNDRY_STATE, BOUNDRY_STATE)): 0.0}
    alpha_pi = {(0, (BOUNDRY_STATE, BOUNDRY_STATE)): 0.0}
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
                q = get_jump_transition(aj, aj_1, source_len)
                e = get_emission(target_token, source_token)
                # print k
                # print v, '|', u
                # print aj, '|', aj_1, '=', q
                # print target_token, '|', source_token, '=', e
                p = pi[(k - 1, u)] + q + e
                alpha_p = alpha_pi[(k - 1, u)] + q + e
                # print 'alpha_p', alpha_p
                if alpha_p == float('-inf'):
                    pdb.set_trace()
                if len(arg_pi[(k - 1, u)]) == 0:
                    bt = [u]
                else:
                    bt = [arg_pi[(k - 1, u)], u]
                max_prob_to_bt[p] = bt
                sum_prob_to_bt.append(alpha_p)

            max_bt = max_prob_to_bt[max(max_prob_to_bt)]
            new_pi_key = (k, v)
            pi[new_pi_key] = max(max_prob_to_bt)
            alpha_pi[new_pi_key] = lu.logadd_of_list(sum_prob_to_bt)
            arg_pi[new_pi_key] = max_bt
    max_bt = max_prob_to_bt[max(max_prob_to_bt)]
    max_p = max(max_prob_to_bt)
    max_bt = flatten_backpointers(max_bt)
    return max_bt, max_p, alpha_pi  # returns the best back trace, best path probability, sum of path probabilites


def get_backwards(obs, trelis, alpha_pi, source_len=None):
    n = len(obs) - 1  # index of last word
    beta_pi = {(n, (BOUNDRY_STATE, BOUNDRY_STATE)): 0.0}
    S = alpha_pi[(n, (BOUNDRY_STATE, BOUNDRY_STATE))]  # from line 13 in pseudo code
    p_unigrams = {}
    p_obs = {}
    p_trans = {}
    for k in range(n, 0, -1):
        for v in trelis[k]:
            pb = beta_pi[(k, v)]
            aj = v[0]
            source_token = v[1]
            posterior_unigram_val = beta_pi[(k, v)] + alpha_pi[(k, v)] - S
            p_obs = do_accumilate_posterior_obs(p_obs, obs[k], aj, source_token, posterior_unigram_val)
            p_unigrams = do_append_posterior_unigrams(p_unigrams, k, v, posterior_unigram_val)
            for u in trelis[k - 1]:
                # print 'reverse transition', 'k', k, 'u', u, '->', 'v', v
                aj_1 = u[0]
                q = get_jump_transition(aj, aj_1, source_len)
                target_token = obs[k]
                e = get_emission(target_token, source_token)
                p = q + e
                beta_p = pb + p
                new_pi_key = (k - 1, u)
                if new_pi_key not in beta_pi:  # implements lines 16
                    beta_pi[new_pi_key] = beta_p
                else:
                    beta_pi[new_pi_key] = lu.logadd(beta_pi[new_pi_key], beta_p)
                posterior_bigram_val = alpha_pi[(k - 1, u)] + p + beta_pi[(k, v)] - S
                p_trans = do_accumilate_posterior_bigrams_jump(p_trans, aj, aj_1, posterior_bigram_val, source_len)

    return p_unigrams, p_trans, p_obs, S, beta_pi


def format_alignments(init_aligns):
    aligns = {}
    for line in init_aligns:
        [snum, inum, jnum] = line.split()
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
        A.append(a + [BOUNDRY_STATE])
    return A


def get_jump_mle(alignments_split, source_split):
    jcounts = {}
    for a, s in zip(alignments_split, source_split):
        alignment_bigrams = [(a[i], a[i - 1]) for i in range(1, len(a))]
        for j1, j0 in alignment_bigrams:
            jkey = jump_key(j1, j0, len(s))
            jcounts[jkey] = lu.logadd(jcounts.get(jkey, float('-inf')), 0.0)
    return jcounts


def update_jump_alignment_mle(posterior_alignment_counts):
    for key in posterior_alignment_counts:
        if key[0] == COUNT_TYPE:
            jump_counts[key] = posterior_alignment_counts[key]


def update_translation_mle(posterior_emission_counts):
    for f, e in translations_probs:
        try:
            counts_fe = posterior_emission_counts['count_emission', f, e]
            count_e = posterior_emission_counts['any_emission_from', e]
            if count_e == float('-inf'):
                translations_probs[f, e] = float('-inf')
            else:
                translations_probs[f, e] = counts_fe - count_e
        except KeyError:
            pass


def get_translation_mle(init_trans):
    translations_mle = {}
    for line in init_trans:
        [fi, ej, p] = line.split()
        if float(p) != 0.0:
            translations_mle[fi, ej] = log(float(p))
        else:
            translations_mle[fi, ej] = float('-inf')
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
        # return 'data/corpus.en', 'data/corpus.es', 'data/model1.trans', 'data/model1.alignments', 'data/hmm.trans',
        # 'data/hmm.alignments', 'data/dev.en', 'data/dev.es'
        exit()


def accumilate(accumilator, addition):
    for k, val in addition.iteritems():
        if isinstance(val, float):
            s = lu.logadd(accumilator.get(k, float('-inf')), val)
            accumilator[k] = s
        elif isinstance(val, set):
            accumilator[k] = accumilator.get(k, set([]))
            accumilator[k].update(val)
    return accumilator


if __name__ == "__main__":

    source, target, init_translations, init_alignments, save_translations_learned, save_alignment_out, source_alignment_test, target_alignment_test = parseargs(
        sys.argv)

    corpus_source = codecs.open(source, 'r', 'utf-8').readlines()
    corpus_target = codecs.open(target, 'r', 'utf-8').readlines()
    dev_source = codecs.open(source_alignment_test, 'r', 'utf-8').readlines()
    dev_target = codecs.open(target_alignment_test, 'r', 'utf-8').readlines()

    z = [(s, t) for s, t in zip(corpus_source, corpus_target) if (s.strip() != '' and t.strip() != '')]
    cs, ct = zip(*z)
    corpus_source = list(cs)
    corpus_target = list(ct)
    alignment_split = format_alignments(codecs.open(init_alignments, 'r', 'utf-8').readlines())
    init_translations = codecs.open(init_translations, 'r', 'utf-8').readlines()

    source_split = [[BOUNDRY_STATE] + i.split() + [BOUNDRY_STATE] for i in corpus_source]
    target_split = [[BOUNDRY_STATE] + i.split() + [BOUNDRY_STATE] for i in corpus_target]
    jump_counts = get_jump_mle(alignment_split, source_split)
    translations_probs = get_translation_mle(init_translations)

    for i in range(3):
        accu_alpha = 0.0
        accu_mu = 0.0
        posterior_transitions_accumilation = {}
        posterior_emission_accumilation = {}
        final_alignments = []
        for idx, (e, f) in enumerate(zip(source_split, target_split)):
            t = get_trellis(f, e)
            max_bt, max_p, alpha_pi = get_viterbi_and_forward(f, t, len(e))
            posterior_uni, posterior_trans, posterior_emission, S, beta_pi = get_backwards(f, t, alpha_pi, len(e))
            accu_alpha += S
            accu_mu += max_p
            sys.stderr.write('iteration: %d sentence %d accumulated alpha %f\n' % (i, idx, accu_alpha))
            sys.stderr.flush()
            if accu_alpha == float('-inf'):
                pdb.set_trace()
            posterior_transitions_accumilation = accumilate(posterior_transitions_accumilation, posterior_trans)
            posterior_emission_accumilation = accumilate(posterior_emission_accumilation, posterior_emission)
            [out_alignments, out_emissions] = zip(*max_bt)
            final_alignments = final_alignments + list(out_alignments)

        update_translation_mle(posterior_emission_accumilation)
        update_jump_alignment_mle(posterior_transitions_accumilation)
        print 'iteration', i, 'mu', accu_mu, 'alpha', accu_alpha


    # TODO this does not write the alignments for the test files at all!
    dev_source_split = [[BOUNDRY_STATE] + i.split() + [BOUNDRY_STATE] for i in dev_source]
    dev_target_split = [[BOUNDRY_STATE] + i.split() + [BOUNDRY_STATE] for i in dev_target]
    final_alignments = []
    for idx, (e, f) in enumerate(zip(dev_source_split, dev_target_split)):
        t = get_trellis(f, e)
        max_bt, max_p, alpha_pi = get_viterbi_and_forward(f, t, len(e))
        [out_alignments, out_emissions] = zip(*max_bt)
        final_alignments = final_alignments + list(out_alignments)
    writer = open(save_alignment_out, 'w')
    ia = 0
    for aj in final_alignments:
        if aj == '###':
            ia += 1
            w = 1
        else:
            if aj != 0:
                writer.write(str(ia) + ' ' + str(aj) + ' ' + str(w) + '\n')
            w += 1
    writer.flush()
    writer.close()
