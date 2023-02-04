"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

from collections import Counter


def score(key, pred):
    correct_dict = Counter()
    guessed_dict = Counter()
    gold_dict = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = pred[row]

        guessed_dict[guess] += 1
        gold_dict[gold] += 1
        if gold == guess:
            correct_dict[guess] += 1

    prec_micro = 1.0
    if sum(guessed_dict.values()) > 0:
        prec_micro = float(sum(correct_dict.values())) / float(sum(guessed_dict.values()))
    recall_micro = 0.0
    if sum(gold_dict.values()) > 0:
        recall_micro = float(sum(correct_dict.values())) / float(sum(gold_dict.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    # print("Precision (micro): {:.3%}".format(prec_micro))
    # print("   Recall (micro): {:.3%}".format(recall_micro))
    # print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro

def AUC(logits, labels):
    num_right = sum(labels)
    num_total = len(labels)
    num_total_pairs = (num_total - num_right) * num_right

    if num_total_pairs == 0:
        return 0.5

    num_right_pairs = 0
    hit_count = 0
    for label in labels:
        if label == 0:
            num_right_pairs += hit_count
        else:
            hit_count += 1

    return float(num_right_pairs) / num_total_pairs

def print_table(*args, header=''):
    print('=' * 100)
    print(header)
    for tup in zip(*args):
        print('\t'.join(['%.3f' % t for t in tup]))