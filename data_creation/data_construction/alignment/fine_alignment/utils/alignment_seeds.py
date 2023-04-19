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


# Part 1: String Match with sliding window
def string_match_sliding_window(en_subset, episode, window_size=5):
    """
    Generate sub-sentences using sliding window algorithm and check whether these sentences exists in the utterance
    """
    res = {}
    for i, subtitle in enumerate(en_subset):
        subtitle = transformation(subtitle)
        subtitle_tokens = subtitle.strip().split(" ")
        if len(subtitle_tokens) < window_size:
            continue

        subtitle_segments = []
        for j in range(len(subtitle_tokens) - window_size):
            subtitle_segments.append(" ".join(subtitle_tokens[j: j + window_size]))

        for j, (utt, speaker) in enumerate(episode):
            utt = transformation(utt)
            for sub_seg in subtitle_segments:
                if sub_seg in utt:
                    if i not in res:
                        res[i] = set()
                        res[i].add(j)
                    else:
                        res[i].add(j)
    output = {}
    for x in res:
        output[x] = sorted(list(res[x]))

    return output


def exact_match(en_subset, episode):
    res = {}
    for i, subtitle in enumerate(en_subset):
        subtitle = transformation(subtitle)
        if len(subtitle.strip().split(" ")) <= 5:
            continue
        # Exact Match for short sentences
        for j, (utt, speaker) in enumerate(episode):
            utt = transformation(utt)
            if subtitle == utt:
                if i not in res:
                    res[i] = set()
                    res[i].add(j)
                else:
                    res[i].add(j)
    output = {}
    for x in res:
        output[x] = sorted(list(res[x]))

    return output


def generate_alignment_seeds(en_subset, tbbt_episode, window_size):
    """
    Generate {subtitle_id: episode_id} dictionary as the seeds for the next step alignments
    This includes:
        1.Exact Match + Filtering by Index to make sure the accuracy
        2.Substring Match + Filtering
    """
    temp_0 = filter_by_idx(exact_match(en_subset, tbbt_episode))
    temp_1 = filter_by_idx(string_match_sliding_window(en_subset, tbbt_episode, window_size))
    merged_temp = merge_episode_alignment(temp_0, temp_1)
    alignment_seeds = filter_by_idx(merged_temp)
    return alignment_seeds
