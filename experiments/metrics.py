from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import Counter
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment

# """
# Non-Singleton
# """
# def f1(p_num, p_den, r_num, r_den, beta=1):
#     p = 0 if p_den == 0 else p_num / float(p_den)
#     r = 0 if r_den == 0 else r_num / float(r_den)
#     return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)
#
#
# class CorefEvaluator(object):
#     def __init__(self):
#         self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]
#
#     def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
#         # Update Coref Scores
#         for e in self.evaluators:
#             e.update(predicted, gold, mention_to_predicted, mention_to_gold)
#
#     def get_f1(self, name):
#         if name == "macro":
#             return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)
#         elif name == "muc":
#             return self.evaluators[0].get_f1()
#         elif name == "b_cubed":
#             return self.evaluators[1].get_f1()
#         elif name == "ceafe":
#             return self.evaluators[2].get_f1()
#         else:
#             return 0
#
#     def get_recall(self, name):
#         if name == "macro":
#             return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)
#         elif name == "muc":
#             return self.evaluators[0].get_recall()
#         elif name == "b_cubed":
#             return self.evaluators[1].get_recall()
#         elif name == "ceafe":
#             return self.evaluators[2].get_recall()
#         else:
#             return 0
#
#     def get_precision(self, name):
#         if name == "macro":
#             return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)
#         elif name == "muc":
#             return self.evaluators[0].get_precision()
#         elif name == "b_cubed":
#             return self.evaluators[1].get_precision()
#         elif name == "ceafe":
#             return self.evaluators[2].get_precision()
#         else:
#             return 0
#
#     def get_prf(self, name):
#         if name == "macro":
#             return self.get_precision("macro"), self.get_recall("macro"), self.get_f1("macro")
#         elif name == "muc":
#             return self.get_precision("muc"), self.get_recall("muc"), self.get_f1("muc")
#         elif name == "b_cubed":
#             return self.get_precision("b_cubed"), self.get_recall("b_cubed"), self.get_f1("b_cubed")
#         elif name == "ceafe":
#             return self.get_precision("ceafe"), self.get_recall("ceafe"), self.get_f1("ceafe")
#         else:
#             return 0, 0, 0
#
# class Evaluator(object):
#     def __init__(self, metric, beta=1):
#         self.p_num = 0
#         self.p_den = 0
#         self.r_num = 0
#         self.r_den = 0
#         self.metric = metric
#         self.beta = beta
#
#     def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
#         if self.metric == ceafe:
#             pn, pd, rn, rd = self.metric(predicted, gold)
#         else:
#             pn, pd = self.metric(predicted, mention_to_gold)
#             rn, rd = self.metric(gold, mention_to_predicted)
#         self.p_num += pn
#         self.p_den += pd
#         self.r_num += rn
#         self.r_den += rd
#
#     def get_f1(self):
#         return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)
#
#     def get_recall(self):
#         return 0 if self.r_num == 0 else self.r_num / float(self.r_den)
#
#     def get_precision(self):
#         return 0 if self.p_num == 0 else self.p_num / float(self.p_den)
#
#     def get_prf(self):
#         return self.get_precision(), self.get_recall(), self.get_f1()
#
#     def get_counts(self):
#         return self.p_num, self.p_den, self.r_num, self.r_den
#
#
# def evaluate_documents(documents, metric, beta=1):
#     evaluator = Evaluator(metric, beta=beta)
#     for document in documents:
#         evaluator.update(document)
#     return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()
#
#
# def b_cubed(clusters, mention_to_gold):
#     num, dem = 0, 0
#
#     for c in clusters:
#         # No Singleton Choice
#         if len(c) == 1:
#             continue
#
#         gold_counts = Counter()
#         correct = 0
#         for m in c:
#             if m in mention_to_gold:
#                 gold_counts[tuple(mention_to_gold[m])] += 1
#         for c2, count in gold_counts.items():
#             # correct += count * count
#             if len(c2) != 1:
#                 correct += count * count
#
#         num += correct / float(len(c))
#         dem += len(c)
#
#     return num, dem
#
#
# def muc(clusters, mention_to_gold):
#     tp, p = 0, 0
#     for c in clusters:
#         p += len(c) - 1
#         tp += len(c)
#         linked = set()
#         for m in c:
#             if m in mention_to_gold:
#                 linked.add(mention_to_gold[m])
#             else:
#                 tp -= 1
#         tp -= len(linked)
#     return tp, p
#
#
# def phi4(c1, c2):
#     return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))
#
#
# def ceafe(clusters, gold_clusters):
#     # # No Singleton Choice
#     clusters = [c for c in clusters if len(c) != 1]
#     scores = np.zeros((len(gold_clusters), len(clusters)))
#     for i in range(len(gold_clusters)):
#         for j in range(len(clusters)):
#             scores[i, j] = phi4(gold_clusters[i], clusters[j])
#     matching = linear_assignment(-scores)
#     # matching2 = linear_sum_assignment(-scores)
#     # matching2 = np.transpose(np.asarray(matching2))
#     similarity = sum(scores[matching[:, 0], matching[:, 1]])
#     return similarity, len(clusters), similarity, len(gold_clusters)
#
#
# def lea(clusters, mention_to_gold):
#     num, dem = 0, 0
#
#     for c in clusters:
#         # No Singleton Choice
#         if len(c) == 1:
#             continue
#
#         common_links = 0
#         all_links = len(c) * (len(c) - 1) / 2.0
#         for i, m in enumerate(c):
#             if m in mention_to_gold:
#                 for m2 in c[i + 1:]:
#                     if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
#                         common_links += 1
#
#         num += len(c) * common_links / float(all_links)
#         dem += len(c)
#
#     return num, dem
#
# class MentionScorer(object):
#     def __init__(self):
#         self._num_gold_mentions = 0
#         self._num_recalled_mentions = 0
#         self._num_predicted_mentions = 0
#
#     def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
#         predicted_mentions = []
#         for cluster in predicted:
#             predicted_mentions.extend(cluster)
#         predicted_mentions = set(predicted_mentions)
#
#         gold_mentions = []
#         for cluster in gold:
#             gold_mentions.extend(cluster)
#         gold_mentions = set(gold_mentions)
#
#         recalled_mentions = gold_mentions & predicted_mentions
#
#         self._num_gold_mentions += len(gold_mentions)
#         self._num_predicted_mentions += len(predicted_mentions)
#         self._num_recalled_mentions += len(recalled_mentions)
#
#     def reset(self):
#         self._num_gold_mentions = 0
#         self._num_recalled_mentions = 0
#         self._num_predicted_mentions = 0
#
#     def get_metric(self, reset: bool = False) -> float:
#         # Calculate Recall
#         if self._num_gold_mentions == 0:
#             recall = 0.0
#         else:
#             recall = self._num_recalled_mentions / self._num_gold_mentions
#
#         # Calculate Precision
#         if self._num_recalled_mentions == 0:
#             precision = 0.0
#         else:
#             precision = self._num_recalled_mentions / self._num_predicted_mentions
#
#         # Calculate F1
#         if precision + recall == 0:
#             f1 = 0.0
#         else:
#             f1 = (2 * precision * recall) / (precision + recall)
#         if reset:
#             self.reset()
#         return precision, recall, f1


"""
Singleton
"""
def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        # Update Coref Scores
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self, name):
        if name == "macro":
            return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)
        elif name == "muc":
            return self.evaluators[0].get_f1()
        elif name == "b_cubed":
            return self.evaluators[1].get_f1()
        elif name == "ceafe":
            return self.evaluators[2].get_f1()
        else:
            return 0

    def get_recall(self, name):
        if name == "macro":
            return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)
        elif name == "muc":
            return self.evaluators[0].get_recall()
        elif name == "b_cubed":
            return self.evaluators[1].get_recall()
        elif name == "ceafe":
            return self.evaluators[2].get_recall()
        else:
            return 0

    def get_precision(self, name):
        if name == "macro":
            return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)
        elif name == "muc":
            return self.evaluators[0].get_precision()
        elif name == "b_cubed":
            return self.evaluators[1].get_precision()
        elif name == "ceafe":
            return self.evaluators[2].get_precision()
        else:
            return 0

    def get_prf(self, name):
        if name == "macro":
            return self.get_precision("macro"), self.get_recall("macro"), self.get_f1("macro")
        elif name == "muc":
            return self.get_precision("muc"), self.get_recall("muc"), self.get_f1("muc")
        elif name == "b_cubed":
            return self.get_precision("b_cubed"), self.get_recall("b_cubed"), self.get_f1("b_cubed")
        elif name == "ceafe":
            return self.get_precision("ceafe"), self.get_recall("ceafe"), self.get_f1("ceafe")
        else:
            return 0, 0, 0

class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        # No Singleton Choice
        # if len(c) == 1:
        #     continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            correct += count * count
            # if len(c2) != 1:
            #     correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    # # No Singleton Choice
    # clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    # matching2 = linear_sum_assignment(-scores)
    # matching2 = np.transpose(np.asarray(matching2))
    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        # No Singleton Choice
        # if len(c) == 1:
        #     continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem

class MentionScorer(object):
    def __init__(self):
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
        self._num_predicted_mentions = 0

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        predicted_mentions = []
        for cluster in predicted:
            predicted_mentions.extend(cluster)
        predicted_mentions = set(predicted_mentions)

        gold_mentions = []
        for cluster in gold:
            gold_mentions.extend(cluster)
        gold_mentions = set(gold_mentions)

        recalled_mentions = gold_mentions & predicted_mentions

        self._num_gold_mentions += len(gold_mentions)
        self._num_predicted_mentions += len(predicted_mentions)
        self._num_recalled_mentions += len(recalled_mentions)

    def reset(self):
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
        self._num_predicted_mentions = 0

    def get_metric(self, reset: bool = False) -> float:
        # Calculate Recall
        if self._num_gold_mentions == 0:
            recall = 0.0
        else:
            recall = self._num_recalled_mentions / self._num_gold_mentions

        # Calculate Precision
        if self._num_recalled_mentions == 0:
            precision = 0.0
        else:
            precision = self._num_recalled_mentions / self._num_predicted_mentions

        # Calculate F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        if reset:
            self.reset()
        return precision, recall, f1



