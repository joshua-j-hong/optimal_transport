#!/usr/bin/env python3

import argparse
import itertools
from collections import Counter


PUNCTUATION_MARKS = {".", ",", "!", "?", ";", ":", "(", ")"}


def parse_args():
    parser = argparse.ArgumentParser("Calculates Alignment Error Rate, output format: AER (Precision, Recall, Alginment-Links-Hypothesis)")
    parser.add_argument("reference", help="path of reference alignment, e.g. '10-9 11p42'")
    parser.add_argument("hypothesis", help="path to hypothesis alignment")

    parser.add_argument("--reverseRef", help="reverse reference alignment", action='store_true')
    parser.add_argument("--reverseHyp", help="reverse hypothesis alignment", action='store_true')

    parser.add_argument("--oneRef", help="reference indices start at index 1", action='store_true')
    parser.add_argument("--oneHyp", help="hypothesis indices start at index 1", action='store_true')
    parser.add_argument("--allSure", help="treat all alignments in the reference as sure alignments", action='store_true')
    parser.add_argument("--ignorePossible", help="Ignore all possible links", action='store_true')
    parser.add_argument("--fAlpha", help="alpha parameter used to calculate f measure (has to be set to a value >= 0.0 to report the f-measure)", default=-1.0, type=float)

    parser.add_argument("--source", default="", help="the source sentence, used for an error analysis")
    parser.add_argument("--target", default="", help="the target sentence, used for an error analysis")
    parser.add_argument("--cleanPunctuation", action="store_true", help="Removes alignments including punctuation marks, that are not aligned to the same punctuation mark (e.g. ','-'that')")
    parser.add_argument("--most_common_errors", default=10, type=int)

    return parser.parse_args()

def calculate_precision_recall_f1(true_positive, false_negative, false_positive, f_alpha):
    if true_positive +  false_positive > 0:
        precision = true_positive / (true_positive +  false_positive)
    else:
        precision = 0
    if true_positive +  false_negative > 0:
        recall = true_positive / (true_positive +  false_negative)
    else:
        recall = 0

    if f_alpha < 0.0:
        f_measure = 0.0
    elif precision == 0 or recall == 0:
        f_measure = 0
    else:
        f_divident = f_alpha / precision
        f_divident += (1.0 - f_alpha) / recall
        f_measure = 1.0 / f_divident
    
    return precision, recall, f_measure

def calculate_null_alignments(gold_labels, predicted, source, target):
    """ Count the number of words with no alignments and how many are captured by the predicted alignments
    """
    gold_source = set(x[0] for x in gold_labels)
    gold_target = set(x[1] for x in gold_labels)
    predicted_source = set(x[0] for x in predicted)
    predicted_target = set(x[1] for x in predicted)

    gold_null_source_indices = set(range(len(source))) - gold_source
    gold_null_target_indices = set(range(len(target))) - gold_target

    pred_null_source_indices = set(range(len(source))) - predicted_source
    pred_null_target_indices = set(range(len(target))) - predicted_target

    true_positive = len(gold_null_source_indices & pred_null_source_indices) + len(gold_null_target_indices & pred_null_target_indices)
    false_negative = len(gold_null_source_indices - pred_null_source_indices) + len(gold_null_target_indices - pred_null_target_indices)
    false_positive = len(pred_null_source_indices - gold_null_source_indices) + len(pred_null_target_indices - gold_null_target_indices)

    return true_positive, false_negative, false_positive

def calculate_one_to_one_alignments(gold_labels, predicted, source, target):
    """ Count the number of words with exactly one alignment and how many are captured by the predicted alignments
    """

    gold_source_counts = dict(Counter(x[0] for x in gold_labels))
    gold_source_target = dict(Counter(x[1] for x in gold_labels))
    gold_one2one_indices = set()
    for source, target in gold_labels:
        if gold_source_counts[source] == 1 and gold_source_target[target] == 1:
            gold_one2one_indices.add((source, target))

    predicted_source_counts = dict(Counter(x[0] for x in predicted))
    predicted_source_target = dict(Counter(x[1] for x in predicted))
    predicted_one2one_indices = set()
    for source, target in predicted:
        if predicted_source_counts[source] == 1 and predicted_source_target[target] == 1:
            predicted_one2one_indices.add((source, target))

    true_positive = len(gold_one2one_indices & predicted_one2one_indices)
    false_negative = len(gold_one2one_indices - predicted_one2one_indices)
    false_positive = len(predicted_one2one_indices - gold_one2one_indices)

    return true_positive, false_negative, false_positive

def calculate_many_to_one_alignments(gold_labels, predicted, source, target):
    """ Count the number of words with exactly one alignment and how many are captured by the predicted alignments
    """

    gold_source_counts = dict(Counter(x[0] for x in gold_labels))
    gold_source_target = dict(Counter(x[1] for x in gold_labels))
    gold_many2one_indices = set()
    for source, target in gold_labels:
        if ((gold_source_counts[source] > 1 or gold_source_target[target] > 1)):
            gold_many2one_indices.add((source, target))

    predicted_source_counts = dict(Counter(x[0] for x in predicted))
    predicted_source_target = dict(Counter(x[1] for x in predicted))
    predicted_many2one_indices = set()
    for source, target in predicted:
        if (predicted_source_counts[source] > 1 or predicted_source_target[target] > 1):
            predicted_many2one_indices.add((source, target))

    true_positive = len(gold_many2one_indices & predicted_many2one_indices)
    false_negative = len(gold_many2one_indices - predicted_many2one_indices)
    false_positive = len(predicted_many2one_indices - gold_many2one_indices)

    return true_positive, false_negative, false_positive

def calculate_internal_jumps(alignments):
    """ Count number of times the set of source word indices aligned to a target word index are not adjacent
        Each non adjacent set of source word indices counts only once
    >>> calculate_internal_jumps([{1,2,4}, {42}])
    1
    >>> calculate_internal_jumps([{1,2,3,4}])
    0
    >>> calculate_internal_jumps([set()])
    0
    """
    def contiguous(s):
        if len(s) <= 1:
            return True
        else:
            elements_in_contiguous_set = max(s) - min(s) + 1
            return elements_in_contiguous_set == len(s)

    return [contiguous(s) for s in alignments].count(False)

def calculate_external_jumps(alignments):
    """ Count number of times the (smallest) source index aligned to target word x is not adjacent or identical to any source word index aligned to the next target word index x+1
        Target words which do not have any source word aligned to it are ignored
    >>> calculate_external_jumps([set(), {1,2,4}, {2}, {4}, set()])
    1
    """

    jumps = 0

    for prev, current in zip(alignments, alignments[1:]):
        if len(prev) > 0 and len(current) > 0:
            src = sorted(prev)[0]
            if src in current or src+1 in current or src-1 in current:
                pass
            else:
                jumps += 1
    return jumps


def to_list(A):
    """ converts set of src-tgt alignments to a list containing a set of aligned source word for each target position
    >>> to_list({(2,1)})
    [set(), {2}]
    """
    max_tgt_idx = max({y for x, y in A}) if len(A) > 0 else 0
    lst = [set() for _ in range(max_tgt_idx+1)]
    for x, y in A:
        lst[y].add(x)
    return lst


def calculate_metrics(array_sure, array_possible, array_hypothesis, f_alpha, source_sentences=(), target_sentences=(), clean_punctuation=False):
    """ Calculates precision, recall and alignment error rate as described in "A Systematic Comparison of Various
        Statistical Alignment Models" (https://www.aclweb.org/anthology/J/J03/J03-1002.pdf) in chapter 5


    Args:
        array_sure: array of sure alignment links
        array_possible: array of possible alignment links
        array_hypothesis: array of hypothesis alignment links
    """

    number_of_sentences = len(array_sure)
    assert number_of_sentences == len(array_possible)
    assert number_of_sentences == len(array_hypothesis)

    errors = Counter()

    sum_a_intersect_p, sum_a_intersect_s, sum_s, sum_a, aligned_source_words, aligned_target_words = 6 * [0.0]
    sum_source_words, sum_target_words = map(lambda s: max(1.0, sum(len(x) for x in s)), [source_sentences, target_sentences])
    internal_jumps, external_jumps = 0, 0
    null_true_positive, null_false_positive, null_false_negative = 0, 0, 0
    one2one_true_positive, one2one_false_positive, one2one_false_negative = 0, 0, 0
    many2one_true_positive, many2one_false_positive, many2one_false_negative = 0, 0, 0


    for S, P, A, source, target in itertools.zip_longest(array_sure, array_possible, array_hypothesis, source_sentences, target_sentences):
        if clean_punctuation:
            A = {(s, t) for (s, t) in A if not ((source[s] in PUNCTUATION_MARKS or target[t] in PUNCTUATION_MARKS) and source[s] != target[t])}
        sum_a += len(A)
        sum_s += len(S)
        sum_a_intersect_p += len(A.intersection(P))
        sum_a_intersect_s += len(A.intersection(S))
        aligned_source_words += len({x for x, y in A})
        aligned_target_words += len({y for x, y in A})
        al = to_list(A)
        internal_jumps += calculate_internal_jumps(al)
        external_jumps += calculate_external_jumps(al)

        if source and target:
            for src_pos, tgt_pos in A:
                if not src_pos < len(source):
                    print(source, len(source), src_pos)
                if not tgt_pos < len(target):
                    print(target, len(target), tgt_pos)
                if (src_pos, tgt_pos) not in P:
                    errors[source[src_pos], target[tgt_pos]] += 1
            true_positive, false_negative, false_positive = calculate_null_alignments(S, A, source, target)
            null_true_positive += true_positive
            null_false_positive += false_positive
            null_false_negative += false_negative

            true_positive, false_negative, false_positive = calculate_one_to_one_alignments(S, A, source, target)
            one2one_true_positive += true_positive
            one2one_false_positive += false_positive
            one2one_false_negative += false_negative

            true_positive, false_negative, false_positive = calculate_many_to_one_alignments(S, A, source, target)
            many2one_true_positive += true_positive
            many2one_false_positive += false_positive
            many2one_false_negative += false_negative
            
    if sum_a > 0:
        precision = sum_a_intersect_p / sum_a
    else:
        precision = 0
    if sum_s > 0:
        recall = sum_a_intersect_s / sum_s
    else:
        recall = 0
    if (sum_a + sum_s) > 0:
        aer = 1.0 - ((sum_a_intersect_p + sum_a_intersect_s) / (sum_a + sum_s))
    else:
        aer = 1.0

    if f_alpha < 0.0:
        f_measure = 0.0
    elif precision == 0 or recall == 0:
        f_measure = 0
    else:
        f_divident = f_alpha / precision
        f_divident += (1.0 - f_alpha) / recall
        f_measure = 1.0 / f_divident

    source_coverage = aligned_source_words / sum_source_words
    target_coverage = aligned_target_words / sum_target_words

    null_precision, null_recall, null_f1 = calculate_precision_recall_f1(null_true_positive, null_false_negative, null_false_positive, f_alpha)
    one2one_precision, one2one_recall, one2one_f1 = calculate_precision_recall_f1(one2one_true_positive, one2one_false_negative, one2one_false_positive, f_alpha)
    many2one_precision, many2one_recall, many2one_f1 = calculate_precision_recall_f1(many2one_true_positive, many2one_false_negative, many2one_false_positive, f_alpha)

    return precision, recall, aer, f_measure, errors, source_coverage, target_coverage, internal_jumps, external_jumps, null_precision, null_recall, null_f1, one2one_precision, one2one_recall, one2one_f1, many2one_precision, many2one_recall, many2one_f1

def parse_single_alignment(string, reverse=False, one_indexed=False):
    assert ('-' in string or 'p' in string) and 'Bad Alignment separator'

    a, b = string.replace('p', '-').split('-')
    a, b = int(a), int(b)

    if one_indexed:
        a = a - 1
        b = b - 1

    #if reverse:
    #    a, b = b, a

    return a, b


def read_text(path):
    if path == "":
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.split() for l in f]


if __name__ == "__main__":
    args = parse_args()
    sure, possible, hypothesis = [], [], []

    source, target = map(read_text, [args.source, args.target])

    assert len(source) == len(target), "Length of source and target does not match"
    assert (not args.cleanPunctuation) or len(source) > 0, "To clean punctuation alignments, specify a source and target text file"
    # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    # print(args.reverseRef) #False
    # print(args.oneRef) #True
    # print(args.cleanPunctuation) #False
    # print(args.allSure)#False
    with open(args.reference, 'r') as f:
        for line in f:
            sure.append(set())
            possible.append(set())

            for alignment_string in line.split():

                sure_alignment = True if '-' in alignment_string else False
                alignment_tuple = parse_single_alignment(alignment_string, args.reverseRef, args.oneRef)

                if sure_alignment or args.allSure:
                    sure[-1].add(alignment_tuple)
                if sure_alignment or not args.ignorePossible:
                    possible[-1].add(alignment_tuple)
    with open(args.hypothesis, 'r') as f:
        for line in f:
            hypothesis.append(set())

            for alignment_string in line.split():
                alignment_tuple = parse_single_alignment(alignment_string, args.reverseHyp, args.oneHyp)
                hypothesis[-1].add(alignment_tuple)

    precision, recall, aer, f_measure, errors, source_coverage, target_coverage, internal_jumps, external_jumps, null_precision, null_recall, null_f1, one2one_precision, one2one_recall, one2one_f1, many2one_precision, many2one_recall, many2one_f1 = calculate_metrics(sure, possible, hypothesis, args.fAlpha, source, target, args.cleanPunctuation)
    print("{0}: {1:.1f}% ({2:.1f}%/{3:.1f}%/{4})".format(args.hypothesis,
                aer * 100.0, precision * 100.0, recall * 100.0, sum([len(x) for x in hypothesis])))
    #print("=======aer========",aer * 100.0)
    if args.fAlpha >= 0.0:
        print("F-Measure: {:.3f}".format(f_measure))

    if args.source:
        assert args.target and args.most_common_errors > 0, "To output the most common errors, define a source and target file and the number of errors to output"
        print(errors.most_common(args.most_common_errors))
        print("Internal Jumps: {}, External Jumps: {}".format(internal_jumps, external_jumps))
        print("Source Coverage: {:.1f}%, Target Coverage: {:.1f}%".format(source_coverage * 100.0, target_coverage * 100.0))
        print("Null Alignment Statistics: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(null_precision, null_recall, null_f1))
        print("One-to-One Alignment Statistics: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(one2one_precision, one2one_recall, one2one_f1))
        print("Many-to-One Alignment Statistics: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(many2one_precision, many2one_recall, many2one_f1))
