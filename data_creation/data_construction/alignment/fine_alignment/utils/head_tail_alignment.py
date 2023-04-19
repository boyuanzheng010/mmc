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


def get_optimal_cer_from_episode(ground_truth, hypothesis_pool, utt_ids):
    scores = []
    for i, hypothesis in enumerate(hypothesis_pool):
        scores.append(jiwer.cer(ground_truth, hypothesis))
    return min(scores), hypothesis_pool[scores.index(min(scores))], ground_truth, utt_ids[scores.index(min(scores))]


def fetch_before_after(en_subset, zh_subset, tbbt_episode, epi2sub):
    gap_pairs = []
    # Fetch the Episode-Subtitle Pair before the episode start
    epi_ids = list(range(0, min(epi2sub.keys())))
    num_epi_token = 0
    for i in epi_ids:
        num_epi_token += len(tbbt_episode[i][0].strip().split(" "))

    sub_ids = [min(epi2sub[min(epi2sub.keys())])]
    num_sub_token = len(en_subset[sub_ids[-1]].strip().split(" "))
    while num_sub_token <= num_epi_token*3:
        sub_ids.append(sub_ids[-1]-1)
        num_sub_token += len(en_subset[sub_ids[-1]].strip().split(" "))
    gap_pairs.append([epi_ids, sorted(sub_ids)])

    # Fetch the Episode-Subtitle Pair after the episode end
    epi_ids = list(range(max(epi2sub.keys())+1, len(tbbt_episode)))
    num_epi_token = 0
    for i in epi_ids:
        num_epi_token += len(tbbt_episode[i][0].strip().split(" "))

    sub_ids = [max(epi2sub[max(epi2sub.keys())])+1]
    num_sub_token = len(en_subset[sub_ids[-1]].strip().split(" "))
    while num_sub_token <= num_epi_token*3:
        sub_ids.append(sub_ids[-1]+1)
        num_sub_token += len(en_subset[sub_ids[-1]].strip().split(" "))
    gap_pairs.append([epi_ids, sub_ids])

    return gap_pairs


def before_after_wer_match(en_subset, episode, epi_ids, sub_ids):
    # Load Sub2Epi
    temp_sub2epi = {}
    temp_epi2sub = {}
    for sub_id in sub_ids:
        subtitle = transformation(en_subset[sub_id].replace("’", " ").replace('…', " "))
        subtitle_tokens = subtitle.strip().split(" ")
        sub_len = len(subtitle_tokens)
        if sub_len <= 4:
            continue

        utt_segments = []
        utt_ids = []
        for epi_id in epi_ids:
            utt = transformation(episode[epi_id][0].replace("’", " ").replace('…', " "))
            utt_tokens = utt.strip().split(" ")
            for j in range(len(utt_tokens) - sub_len+1):
                utt_segments.append(" ".join(utt_tokens[j: j + sub_len]))
                utt_ids.append(epi_id)
        if utt_segments != [] and subtitle not in [" ", ""]:
            score, hypo, truth, index = get_optimal_cer_from_episode(subtitle, utt_segments, utt_ids)
            if score >= 0.5:
                continue
            # print("**********")
            # print("Score:", score)
            # print("Subtitle:", truth)
            # print("Episode:", hypo)
            # print("Epi ID:", index, "Sub ID:", sub_id)
            temp_sub2epi[sub_id] = [index]
            if index not in temp_epi2sub:
                temp_epi2sub[index] = [sub_id]
            else:
                temp_epi2sub[index].append(sub_id)
    # print("Sub2Epi:", temp_sub2epi)
    # print("Epi2Sub:", temp_epi2sub)
    return temp_sub2epi

















