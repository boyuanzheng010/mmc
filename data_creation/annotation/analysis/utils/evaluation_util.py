import csv
import json
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Any, Dict, List, Tuple
from collections import Counter


def label_mention_to_cluster(instance, golden_clusters):
    """
    Assign cluster id to mention query
    If answer is notMention, label==-1
    If answer is singleton, label==-2
    Else, label==cluster_id
    """
    mentions = instance['answers']
    labels = []
    for query, answer in mentions:
        # -1: notMention, -2: Singleton, index: cluster_id
        # If it is not mention
        if answer == "notMention":
            labels.append(-1)
        else:
            idx = -2
            for i, cluster in enumerate(golden_clusters):
                if (answer[0] in cluster) and (len(cluster) != 1):
                    idx = i
            labels.append(idx)
    return labels


def kappa(instance1, instance2):
    golden_clusters = instance1['clusters']
    label1 = label_mention_to_cluster(instance1, golden_clusters)
    label2 = label_mention_to_cluster(instance2, golden_clusters)
    return cohen_kappa_score(label1, label2)


def exact_match(instance1, instance2):
    golden_clusters = instance1['clusters']
    label1 = label_mention_to_cluster(instance1, golden_clusters)
    label2 = label_mention_to_cluster(instance2, golden_clusters)
    return accuracy_score(label1, label2)


# def kappa(instance1, instance2):
#     return cohen_kappa_score(instance1['answers'], instance2['answers'])
#
#
# def exact_match(instance1, instance2):
#     return accuracy_score(instance1['answers'], instance2['answers'])


def b_cubed(instance1, instance2):
    """
    Averaged per-mention precision and recall.
    <https://pdfs.semanticscholar.org/cfe3/c24695f1c14b78a5b8e95bcbd1c666140fd1.pdf>
    """
    clusters, mention_to_gold = instance1['clusters'], instance2['clusters']
    numerator, denominator = 0, 0
    for cluster in clusters:
        if len(cluster) == 1:
            continue
        gold_counts = Counter()
        correct = 0
        for mention in cluster:
            if mention in mention_to_gold:
                gold_counts[tuple(mention_to_gold[mention])] += 1
        for cluster2, count in gold_counts.items():
            if len(cluster2) != 1:
                correct += count * count
        numerator += correct / float(len(cluster))
        denominator += len(cluster)
    return numerator, denominator


def muc(instance1, instance2):
    """
    Counts the mentions in each predicted cluster which need to be re-allocated in
    order for each predicted cluster to be contained by the respective gold cluster.
    <https://aclweb.org/anthology/M/M95/M95-1005.pdf>
    """
    clusters, mention_to_gold = instance1['clusters'], instance2['clusters']
    true_p, all_p = 0, 0
    for cluster in clusters:
        all_p += len(cluster) - 1
        true_p += len(cluster)
        linked = set()
        for mention in cluster:
            if mention in mention_to_gold:
                linked.add(mention_to_gold[mention])
            else:
                true_p -= 1
        true_p -= len(linked)
    return true_p, all_p


def phi4(gold_clustering, predicted_clustering):
    """
    Subroutine for ceafe. Computes the mention F measure between gold and
    predicted mentions in a cluster.
    """
    return (
            2
            * len([mention for mention in gold_clustering if mention in predicted_clustering])
            / (len(gold_clustering) + len(predicted_clustering))
    )


def ceafe(instance1, instance2):
    """
    Computes the Constrained Entity-Alignment F-Measure (CEAF) for evaluating coreference.
    Gold and predicted mentions are aligned into clusterings which maximise a metric - in
    this case, the F measure between gold and predicted clusters.
    <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
    """
    clusters = instance1['clusters']
    gold_clusters = instance2['clusters']

    clusters = [cluster for cluster in clusters if len(cluster) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i, gold_cluster in enumerate(gold_clusters):
        for j, cluster in enumerate(clusters):
            scores[i, j] = phi4(gold_cluster, cluster)
    row, col = linear_sum_assignment(-scores)
    similarity = sum(scores[row, col])
    return similarity, len(clusters), similarity, len(gold_clusters)
