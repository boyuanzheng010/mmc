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


def add_neighbor_subtitles_to_episode(en_subset, epi2sub_alignment_2, episode):
    """
    Iterate every episode utterance (epi_id) and its corresponding subtitle (sub_id)
    Check whether the subtitle before all aligned subtitle and after all aligned subtitle could be added to the current episode
    """
    temp = {}
    for epi_id in epi2sub_alignment_2:
        # Define episode utterance id
        epi_id_former = epi_id - 1
        epi_id = epi_id
        epi_id_latter = epi_id + 1

        # Define subtitle id
        sub_id_former = min(epi2sub_alignment_2[epi_id]) - 1
        sub_id_latter = max(epi2sub_alignment_2[epi_id]) + 1
        sub_ids = sorted(list(epi2sub_alignment_2[epi_id]))

        # Check whether subtitle nearby is in the utterance
        sub_former = transformation(en_subset[sub_id_former])
        sub_latter = transformation(en_subset[sub_id_latter])
        epi = transformation(episode[epi_id][0])

        if sub_former in epi:
            sub_ids.append(sub_id_former)
        if sub_latter in epi:
            sub_ids.append(sub_id_latter)
        temp[epi_id] = sorted(sub_ids)

    return temp


def add_strict_match_within_gaps(gaps, epi2sub, en_subset, tbbt_episode):
    for gap in gaps:
        sub_ids = gap[1]
        epi_ids = gap[0]

        for sub_id in sub_ids:
            sub = transformation(en_subset[sub_id].replace("’", "'"))
            for epi_id in epi_ids:
                try:
                    epi = transformation(tbbt_episode[epi_id][0].replace("’", "'"))
                except:
                    print(epi_id, len(tbbt_episode))
                if len(epi.strip().split(" ")) <= 2:
                    continue
                if sub == epi:
                    if epi_id not in epi2sub:
                        epi2sub[epi_id] = [sub_id]
                    else:
                        epi2sub[epi_id].append(sub_id)

    output = {}
    for epi_id in sorted(list(epi2sub.keys())):
        output[epi_id] = sorted(list(set(epi2sub[epi_id])))

    return output


def add_wer_match_within_gaps(gaps, epi2sub, en_subset, tbbt_episode):
    count = 0
    for gap in gaps:
        sub_ids = gap[1]
        epi_ids = gap[0]

        for epi_id in epi_ids:
            best_score = 100
            best_pair = [None, None]
            epi = transformation(tbbt_episode[epi_id][0].replace("’", "'"))
            if len(epi.strip().split(" ")) <= 2:
                continue
            for sub_id in sub_ids:
                sub = transformation(en_subset[sub_id].replace("’", "'"))
                score = jiwer.compute_measures(epi, sub)['wer']
                if score < best_score:
                    best_score = score
                    best_pair = [epi_id, sub_id]
            if best_score < 0.15:
                count += 1
                if best_pair[0] not in epi2sub:
                    epi2sub[best_pair[0]] = [best_pair[1]]
                else:
                    epi2sub[best_pair[0]].append([best_pair[1]])
    output = {}
    for epi_id in sorted(list(epi2sub.keys())):
        output[epi_id] = sorted(list(set(epi2sub[epi_id])))

    return output


def add_wer_substring_match_within_gaps(gaps, epi2sub, en_subset, tbbt_episode):
    count = 0
    pattern = r'\.|\?|\!|\;|- '
    temp_sub2epi = {}
    for gap in gaps:
        # Build substrings
        sub_ids = gap[1]
        epi_ids = gap[0]
        sub_lists = []
        epi_lists = []

        for epi_id in epi_ids:
            epi = tbbt_episode[epi_id][0].replace("’", "'")
            epi_substring = re.split(pattern, epi)
            for item in epi_substring:
                if len(item.strip().split(" ")) <= 2:
                    continue
                temp_item = transformation(item.replace("-", " ").strip())
                epi_lists.append([temp_item, epi_id])

        for sub_id in sub_ids:
            sub = en_subset[sub_id].replace("’", "'")
            sub_substring = re.split(pattern, sub)
            for item in sub_substring:
                if len(item.strip().split(" ")) <= 2:
                    continue
                temp_item = transformation(item.replace("-", " ").strip())
                sub_lists.append([temp_item, sub_id])

        # Calculate WER Similarity
        for (sub, sub_id) in sub_lists:
            for (epi, epi_id) in epi_lists:
                cer = jiwer.cer(epi, sub)
                if cer <= 0.2:
                    count += 1
                    if sub_id not in temp_sub2epi:
                        temp_sub2epi[sub_id] = set()
                        temp_sub2epi[sub_id].add(epi_id)
                    else:
                        temp_sub2epi[sub_id].add(epi_id)

    for sub_id in temp_sub2epi:
        epi_ids = list(temp_sub2epi[sub_id])
        if len(epi_ids) != 1:
            continue
        epi_id = epi_ids[0]
        if epi_id not in epi2sub:
            epi2sub[epi_id] = [sub_id]
        else:
            epi2sub[epi_id].append(sub_id)

    output = {}
    for epi_id in sorted(list(epi2sub.keys())):
        output[epi_id] = sorted(list(set(epi2sub[epi_id])))
    return output


def get_final_stage_gap_pairs(epi2sub):
    # Gather the gap of subtitle corresponding to episode utterance
    epi_keys = sorted(list(epi2sub.keys()))
    sub_keys = sorted(list(turn_sub2epi_into_epi2sub(epi2sub).keys()))

    subtitle_gaps = {}
    for i in range(len(epi_keys) - 1):
        epi_start = epi_keys[i]
        epi_end = epi_keys[i + 1]
        key = (epi_start, epi_end)
        if max(epi2sub[epi_start]) + 1 < min(epi2sub[epi_end]):
            subtitle_gaps[(epi_start, epi_end)] = [item for item in
                                                   range(max(epi2sub[epi_start]) + 1, min(epi2sub[epi_end]))]

    # Perform string match and CER Scoring
    return subtitle_gaps


def get_sliding_window_substrings(input_string, window_size):
    input_tokens = input_string.strip().split(" ")
    substrings = []
    for i in range(len(input_tokens) - window_size):
        substrings.append(" ".join(input_tokens[i: i + window_size]))
    return substrings


def extend_neighbors_episode_sliding(en_subset, epi2sub, tbbt_episode):
    """
    Explore the neighbor subtitles of a episode.

    Given a episode utterance (epi_id), then we fetch the unaligned subtitle (sub_id)
    [epi_id_0, epi_id_1, etc., epi_id_n] - [sub_id_0, sub_id_1, etc. sub_id_m]

    Then, we search within the subset-pair

    For each subtitle, we use sliding window to fetch a set of substrings in each episode utterance and calculate the CER
    """
    # Gather the gap of subtitle corresponding to episode utterance
    subtitle_gaps = get_final_stage_gap_pairs(epi2sub)

    # Iterate the whole subtitle gaps to perform substring match
    temp_epi2sub = deepcopy(epi2sub)
    for item in subtitle_gaps:
        epi_ids = [i for i in range(item[0], item[1] + 1)]
        sub_ids = [i for i in subtitle_gaps[item]]

        # Fetch all episodes and subtitles
        epis = [transformation(tbbt_episode[i][0].replace("’", " ").replace('…', " ")) for i in epi_ids]
        subs = [transformation(en_subset[i].replace("’", " ").replace('…', " ")) for i in sub_ids]

        for sub_id, sub in zip(sub_ids, subs):
            sub_len = len(sub.strip().split(" "))
            if sub_len <= 3:
                continue
            min_score = float('inf')
            min_substring = ""
            source_episode = ""
            source_sub_id = float('inf')
            source_epi_id = float('inf')
            for epi, epi_id in zip(epis, epi_ids):
                # min_score = float('inf')
                # min_substring = ""
                epi_substrings = get_sliding_window_substrings(epi, window_size=sub_len)
                for substring in epi_substrings:
                    wer = jiwer.wer(sub, substring)
                    if wer <= min_score:
                        min_score = wer
                        min_substring = substring
                        source_episode = epi
                        source_epi_id = epi_id
            if min_score <= 0.5:
                if source_epi_id not in temp_epi2sub:
                    temp_epi2sub[source_epi_id] = [sub_id]
                else:
                    temp_epi2sub[source_epi_id].append(sub_id)
    output = {}
    for epi_id in sorted(list(temp_epi2sub.keys())):
        output[epi_id] = sorted(list(set(temp_epi2sub[epi_id])))
    return output
