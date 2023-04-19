import pickle as pkl
import json
from collections import defaultdict
import jiwer
from copy import deepcopy
import re
import string
from .helper_functions import *

# Set the data cleaning method
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.Strip()
])



def remove_patterns(string):
    """
    Remove the pattern of [ABCDEFG] and NAME:
    """
    pattern_1 = u"\\(.*?\\)|\\[.*?]"
    string_1 = re.sub(pattern_1, "", string)
    pattern_2 = r'[A-Z]+\:'
    string_2 = re.sub(pattern_2, "", string_1)
    return string_2

def get_substrings(string, length):
    substrings = []
    string_tokens = transformation(string.replace("’", "'").replace('…', " ")).strip().split(" ")
    for k in range(len(string_tokens)-length+1):
        substrings.append(" ".join(string_tokens[k: k+ length]))
    return substrings


def get_optimal_cer(ground_truth, hypothesis_pool):
    scores = []
    for i, hypothesis in enumerate(hypothesis_pool):
        scores.append(jiwer.cer(ground_truth, hypothesis))
    return min(scores), hypothesis_pool[scores.index(min(scores))], ground_truth, scores.index(min(scores))


def get_optimal_wer_from_episode(ground_truth, hypothesis_pool):
    scores = []
    for i, hypothesis in enumerate(hypothesis_pool):
        scores.append(jiwer.compute_measures(ground_truth, hypothesis)['wer'])
    return min(scores), hypothesis_pool[scores.index(min(scores))], ground_truth, scores.index(min(scores))


def ultimate_alignment(gaps, epi2sub, en_subset, tbbt_episode):
    temp = {}
    for (epi_start, epi_end) in gaps:
        epi_ids = list(range(epi_start, epi_end + 1))
        sub_ids = gaps[(epi_start, epi_end)]
        for sub_id in sub_ids:
            sub_sent = transformation(remove_patterns(en_subset[sub_id]))
            sub_tokens = sub_sent.strip().split(" ")

            all_epi_substrings = []
            all_epi_substring_ids = []
            for epi_id in epi_ids:
                epi_sent = transformation(remove_patterns(tbbt_episode[epi_id][0]))
                epi_substrings = get_substrings(epi_sent, len(sub_tokens))
                for item in epi_substrings:
                    all_epi_substrings.append(item)
                    all_epi_substring_ids.append(epi_id)

            if len(all_epi_substrings) != 0 and len(sub_sent) != 0:
                cer, c_substring, c_ground_truth, c_score_idx = get_optimal_cer(sub_sent, all_epi_substrings)

                if cer < 0.55:
                    temp[sub_id] = [all_epi_substring_ids[c_score_idx]]
                else:
                    wer, w_substring, w_ground_truth, w_score_idx = get_optimal_wer_from_episode(sub_sent,
                                                                                                 all_epi_substrings)
                    if wer < 0.5:
                        temp[sub_id] = [all_epi_substring_ids[w_score_idx]]

    sub2epi = turn_sub2epi_into_epi2sub(epi2sub)
    for sub_id in temp:
        sub2epi[sub_id] = temp[sub_id]
    output = {}
    for sub_id in sorted(list(sub2epi.keys())):
        output[sub_id] = sub2epi[sub_id]

    epi2sub = turn_sub2epi_into_epi2sub(output)
    for epi_id in epi2sub:
        sub_ids = epi2sub[epi_id]
        num_epi_tokens = len(transformation(remove_patterns(tbbt_episode[epi_id][0])))
        if len(sub_ids) != max(sub_ids) - min(sub_ids) + 1:
            num_sub_tokens = 0
            for sub_id in range(min(sub_ids), max(sub_ids)):
                num_sub_tokens += len(transformation(remove_patterns(en_subset[sub_id])))
            if abs(num_epi_tokens - num_sub_tokens) / num_epi_tokens * 100 < 100:
                epi2sub[epi_id] = list(range(min(sub_ids), max(sub_ids) + 1))

    return turn_sub2epi_into_epi2sub(output)






